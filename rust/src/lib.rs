// Copyright 2023-present Kensho Technologies, LLC.
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::cmp;
use std::collections::HashSet;

const GAP_VAL: i64 = -1;

/// Computes an optimal global pairwise alignment between two sequences of integers using the
/// Needleman-Wunsch algorithm and returns the corresponding aligned sequences, with any gaps
/// represented by `gap_val`.
///
/// # Notes
/// Unlike other implementations, this only considers a **single backpointer** when backtracing the
/// optimal pairwise alignment, rather than potentially two or three backpointers for each cell if
/// the scores are equal. Rather, this will prioritize "up" transitions (*i.e.*, gap in `seq_two`)
/// over "left" transitions (*i.e.*, gap in `seq_one`), which in turn is prioritized over "diagonal"
/// transitions (*i.e.*, no gap). This is a somewhat arbitrary distinction, but is consistent and
/// leads to a simpler implementation that is both faster and uses less memory.
///
/// # Complexity
/// This takes O(mn) time and O(mn) space complexity, where m and n are the lengths of the two
/// sequences, respectively.
///
/// # References
/// https://en.wikipedia.org/wiki/Needleman%E2%80%93Wunsch_algorithm
#[pyfunction]
#[pyo3(signature = (seq_one, seq_two, match_score=1.0, mismatch_score=-1.0, indel_score=-1.0, gap_val=-1))]
pub fn needleman_wunsch(
    seq_one: Vec<i64>,
    seq_two: Vec<i64>,
    match_score: f64,
    mismatch_score: f64,
    indel_score: f64,
    gap_val: i64,
) -> PyResult<(Vec<i64>, Vec<i64>)> {
    needleman_wunsch_inner(
        seq_one,
        seq_two,
        match_score,
        mismatch_score,
        indel_score,
        gap_val,
    ).map_err(|e| PyValueError::new_err(e.to_string()))
}

pub fn needleman_wunsch_rust(
    seq_one: Vec<&str>,
    seq_two: Vec<&str>,
    match_score: f64,
    mismatch_score: f64,
    indel_score: f64,
    gap: &str,
) -> anyhow::Result<(Vec<String>, Vec<String>)> {
    let (idx2symbol, seq_a_indices, seq_b_indices) = entry2idx(
        &seq_one, &seq_two, gap,
    )?;

    let (seq_a_indices_aligned, seq_b_indices_aligned) = needleman_wunsch_inner(
        seq_a_indices,
        seq_b_indices,
        match_score,
        mismatch_score,
        indel_score,
        GAP_VAL,
    )?;

    Ok(idx2entry(&idx2symbol, &seq_a_indices_aligned, &seq_b_indices_aligned, gap, GAP_VAL))
}

fn needleman_wunsch_inner(
    seq_one: Vec<i64>,
    seq_two: Vec<i64>,
    match_score: f64,
    mismatch_score: f64,
    indel_score: f64,
    gap_val: i64,
) -> anyhow::Result<(Vec<i64>, Vec<i64>)> {
    // Invariant -- gap_val cannot be in either sequence
    if (seq_one.contains(&gap_val)) || (seq_two.contains(&gap_val)) {
        return Err(anyhow::anyhow!(
            "Gap value {gap_val} cannot be present in either sequence",
        ));
    }

    // Use the shorter of the two sequences for the column dimension so that there's less memory
    // fragmentation
    let seq_one_len = seq_one.len();
    let seq_two_len = seq_two.len();
    let (swapped, seq_one_proc, seq_two_proc) = if seq_two_len > seq_one_len {
        (true, seq_two, seq_one)
    } else {
        (false, seq_one, seq_two)
    };
    let seq_one_proc_len = seq_one_proc.len();
    let seq_two_proc_len = seq_two_proc.len();

    let minimum_seq_len = cmp::max(seq_one_proc_len, seq_two_proc_len);
    let mut aligned_seq_one_proc = Vec::<i64>::with_capacity(minimum_seq_len);
    let mut aligned_seq_two_proc = Vec::<i64>::with_capacity(minimum_seq_len);
    if minimum_seq_len == 0 {
        // Both sequences empty -- no alignment needed
        return Ok((aligned_seq_one_proc, aligned_seq_two_proc));
    }

    // Convenience aliases
    let num_rows = seq_one_proc_len + 1;
    let num_cols = seq_two_proc_len + 1;

    // Initialize score matrix with "border" cells marked with indel penalties increasing from the
    // origin
    let mut scores: Vec<f64> = (0..num_rows)
        .flat_map(|row_idx| {
            (0..num_cols)
                .map(|col_idx| {
                    if row_idx == 0 {
                        (col_idx as f64) * indel_score
                    } else if col_idx == 0 {
                        (row_idx as f64) * indel_score
                    } else {
                        0.0
                    }
                })
                .collect::<Vec<f64>>()
        })
        .collect();

    // Initialize backpointers matrix for tracing alignment later with "border" cells pointing to
    // their neighbors, as these are only used if one sequence is fully exhausted before the other.
    // These "backpointers" are simply indices (flattened row-col pairs) into the array. Once one
    // reaches index 0 in backtracing, the sequence is complete.
    let mut backpointers: Vec<usize> = (0..num_rows)
        .flat_map(|row_idx| {
            (0..num_cols)
                .map(|col_idx| {
                    if (row_idx == 0) && (col_idx > 0) {
                        col_idx - 1
                    } else if (col_idx == 0) && (row_idx > 0) {
                        (row_idx - 1) * num_cols
                    } else {
                        0
                    }
                })
                .collect::<Vec<usize>>()
        })
        .collect();

    // Iterate row-by-row, calculating scores for each cell by comparing sequence values at the
    // respective indices to determine if a match or mismatch, then adding an insertion-deletion
    // (indel) score if moving left or up (not diagonally).
    for row_idx in 1..num_rows {
        let seq_one_proc_idx = row_idx - 1;
        for col_idx in 1..num_cols {
            let cell_idx = (row_idx * num_cols) + col_idx;

            let seq_two_proc_idx = col_idx - 1;

            // Check if match or mismatch
            let compare_score = if seq_one_proc[seq_one_proc_idx] == seq_two_proc[seq_two_proc_idx]
            {
                match_score
            } else {
                mismatch_score
            };

            // Now, score transitions from diagonal, up and left, then pick the best
            let diagonal_idx = cell_idx - num_cols - 1;
            let diagonal_score = scores[diagonal_idx] + compare_score;

            let up_idx = cell_idx - num_cols;
            let up_score = scores[up_idx] + indel_score;

            let left_idx = cell_idx - 1;
            let left_score = scores[left_idx] + indel_score;

            let (transition_score, transition_backpointer) =
                if (diagonal_score >= up_score) && (diagonal_score >= left_score) {
                    // Diagonal is the best (or tied)
                    (diagonal_score, diagonal_idx)
                } else if (left_score >= up_score) && (left_score >= diagonal_score) {
                    // Left is the best (or tied)
                    (left_score, left_idx)
                } else {
                    // Up is the best (or tied)
                    (up_score, up_idx)
                };
            scores[cell_idx] = transition_score;
            backpointers[cell_idx] = transition_backpointer;
        }
    }

    // Now, trace back the backpointers to find the optimal sequence, constructing the aligned
    // sequences in the process. Preallocate to the longer of the two sequences, as it will be at
    // least that long no matter what.

    // Start from bottom right corner
    let mut current_backpointer = (num_rows * num_cols) - 1;

    // Follow backpointers
    while current_backpointer > 0 {
        let current_bp_col_idx = current_backpointer % num_cols;
        let current_bp_row_idx = (current_backpointer - current_bp_col_idx) / num_cols;

        let next_backpointer = backpointers[current_backpointer];
        let next_bp_col_idx = next_backpointer % num_cols;
        let next_bp_row_idx = (next_backpointer - next_bp_col_idx) / num_cols;

        if current_bp_row_idx == 0 {
            // Already exhausted sequence A -- add gap
            aligned_seq_one_proc.push(gap_val);
        } else {
            let current_seq_one_proc_idx = current_bp_row_idx - 1;
            if next_bp_row_idx == current_bp_row_idx {
                aligned_seq_one_proc.push(gap_val);
            } else {
                aligned_seq_one_proc.push(seq_one_proc[current_seq_one_proc_idx]);
            }
        }

        if current_bp_col_idx == 0 {
            // Already exhausted sequence B -- add gap
            aligned_seq_two_proc.push(gap_val);
        } else {
            let current_seq_two_proc_idx = current_bp_col_idx - 1;
            if next_bp_col_idx == current_bp_col_idx {
                aligned_seq_two_proc.push(gap_val);
            } else {
                aligned_seq_two_proc.push(seq_two_proc[current_seq_two_proc_idx]);
            }
        }

        current_backpointer = next_backpointer;
    }

    // Reverse sequence, swap back if needed, and return!
    aligned_seq_one_proc.reverse();
    aligned_seq_two_proc.reverse();

    let (aligned_seq_one, aligned_seq_two) = if swapped {
        (aligned_seq_two_proc, aligned_seq_one_proc)
    } else {
        (aligned_seq_one_proc, aligned_seq_two_proc)
    };

    Ok((aligned_seq_one, aligned_seq_two))
}

// See NWScore() subroutine at https://en.wikipedia.org/wiki/Hirschberg%27s_algorithm
// Lower memory if seq_two is the SHORTER (or equal) of the two sequences.
fn nw_score(
    seq_one: &[i64],
    seq_two: &[i64],
    match_score: f64,
    mismatch_score: f64,
    indel_score: f64,
    reverse: bool,
) -> Vec<f64> {
    // Initialize 2-row score matrix with top row cell marked with indel penalties increasing from
    // the origin, as well as first column of bottom row
    let num_cols = seq_two.len() + 1;
    let mut scores: Vec<f64> = (0..(2 * num_cols))
        .map(|idx| {
            if idx <= num_cols {
                (idx as f64) * indel_score
            } else {
                0.0
            }
        })
        .collect();

    // Create copies of the sequences for processing, reversing if needed.
    let seq_one_proc: Vec<i64> = if reverse {
        seq_one.iter().rev().copied().collect()
    } else {
        seq_one.to_vec()
    };

    let seq_two_proc: Vec<i64> = if reverse {
        seq_two.iter().rev().copied().collect()
    } else {
        seq_two.to_vec()
    };

    // Now, fill in scores
    for seq_one_proc_item in &seq_one_proc {
        scores[num_cols] = scores[0] + indel_score;

        for (seq_two_proc_idx, seq_two_proc_item) in seq_two_proc.iter().enumerate() {
            // Check transition scores with the same prioritization as needleman_wunsch()
            let cell_idx = num_cols + seq_two_proc_idx + 1;

            // Check if match or mismatch
            let compare_score = if seq_one_proc_item == seq_two_proc_item {
                match_score
            } else {
                mismatch_score
            };

            // Now, score transitions from diagonal, up and left, then pick the best
            let diagonal_idx = cell_idx - num_cols - 1;
            let diagonal_score = scores[diagonal_idx] + compare_score;

            let up_idx = cell_idx - num_cols;
            let up_score = scores[up_idx] + indel_score;

            let left_idx = cell_idx - 1;
            let left_score = scores[left_idx] + indel_score;

            let transition_score = if (diagonal_score >= up_score) && (diagonal_score >= left_score)
            {
                // Diagonal is the best (or tied)
                diagonal_score
            } else if (left_score >= up_score) && (left_score >= diagonal_score) {
                // Left is the best (or tied)
                left_score
            } else {
                // Up is the best (or tied)
                up_score
            };
            scores[cell_idx] = transition_score;
        }

        // Finally, copy the bottom row to the top row for the next iteration
        for col_idx in 0..num_cols {
            scores[col_idx] = scores[num_cols + col_idx];
        }
    }

    // Return just the final row
    let mut final_scores = Vec::<f64>::with_capacity(num_cols);
    for col_idx in 0..num_cols {
        final_scores.push(scores[num_cols + col_idx]);
    }
    final_scores
}

/// Computes an optimal global pairwise alignment between two sequences of integers using the
/// Hirschberg algorithm (a modification of Needleman-Wunsch that reduces space complexity from
/// O(mn) to O(min(m, n))) and returns the corresponding aligned sequences, with any gaps
/// represented by `gap_val`.
///
/// # Notes
/// Unlike other implementations, this only considers a **single backpointer** when backtracing the
/// optimal pairwise alignment, rather than potentially two or three backpointers for each cell if
/// the scores are equal. Rather, this will prioritize "up" transitions (*i.e.*, gap in `seq_two`)
/// over "left" transitions (*i.e.*, gap in `seq_one`), which in turn is prioritized over "diagonal"
/// transitions (*i.e.*, no gap). This is a somewhat arbitrary distinction, but is consistent and
/// leads to a simpler implementation that is both faster and uses less memory.
///
/// # Complexity
/// This takes O(mn) time and O(min(m, n)) space complexity, where m and n are the lengths of the
/// two sequences, respectively.
///
/// # References
/// https://en.wikipedia.org/wiki/Hirschberg%27s_algorithm
#[pyfunction]
#[pyo3(signature = (seq_one, seq_two, match_score=1.0, mismatch_score=-1.0, indel_score=-1.0, gap_val=-1))]
pub fn hirschberg(
    seq_one: Vec<i64>,
    seq_two: Vec<i64>,
    match_score: f64,
    mismatch_score: f64,
    indel_score: f64,
    gap_val: i64,
) -> PyResult<(Vec<i64>, Vec<i64>)> {
    hirschberg_inner(
        seq_one,
        seq_two,
        match_score,
        mismatch_score,
        indel_score,
        gap_val,
    ).map_err(|e| PyValueError::new_err(e.to_string()))
}

pub fn hirschberg_rust(
    seq_one: Vec<&str>,
    seq_two: Vec<&str>,
    match_score: f64,
    mismatch_score: f64,
    indel_score: f64,
    gap: &str,
) -> anyhow::Result<(Vec<String>, Vec<String>)> {
    let (idx2symbol, seq_a_indices, seq_b_indices) = entry2idx(
        &seq_one, &seq_two, gap,
    )?;

    let (seq_a_indices_aligned, seq_b_indices_aligned) = hirschberg_inner(
        seq_a_indices,
        seq_b_indices,
        match_score,
        mismatch_score,
        indel_score,
        GAP_VAL,
    )?;

    Ok(idx2entry(&idx2symbol, &seq_a_indices_aligned, &seq_b_indices_aligned, gap, GAP_VAL))
}

fn hirschberg_inner(
    seq_one: Vec<i64>,
    seq_two: Vec<i64>,
    match_score: f64,
    mismatch_score: f64,
    indel_score: f64,
    gap_val: i64,
) -> anyhow::Result<(Vec<i64>, Vec<i64>)> {
    // Invariant -- gap_val cannot be in either sequence
    if (seq_one.contains(&gap_val)) || (seq_two.contains(&gap_val)) {
        return Err(anyhow::anyhow!(
            "Gap value {gap_val} cannot be present in either sequence",
        ));
    }

    let seq_one_len = seq_one.len();
    let seq_two_len = seq_two.len();
    if (seq_one_len <= 1) || (seq_two_len <= 1) {
        // Already efficient for these dimensions
        return needleman_wunsch_inner(
            seq_one,
            seq_two,
            match_score,
            mismatch_score,
            indel_score,
            gap_val,
        );
    }

    // First, swap if needed so that the second sequence is the shorter one (uses less memory
    // in nw_score() as a result, as the vector size is only dependent on the second sequence)
    let (swapped, seq_one_proc, seq_two_proc) = if seq_two_len > seq_one_len {
        (true, seq_two, seq_one)
    } else {
        (false, seq_one, seq_two)
    };
    let seq_one_proc_len = seq_one_proc.len();
    let seq_two_proc_len = seq_two_proc.len();

    // Compute last line of Needleman-Wunsch score for both the "left" and "right" partitions
    // of the longer sequence (A)
    let seq_one_proc_mid = seq_one_proc_len / 2;
    let seq_one_proc_l = &seq_one_proc[0..seq_one_proc_mid];
    let seq_one_proc_r = &seq_one_proc[seq_one_proc_mid..seq_one_proc_len];

    let score_l = nw_score(
        seq_one_proc_l,
        &seq_two_proc,
        match_score,
        mismatch_score,
        indel_score,
        false,
    );

    let score_r = nw_score(
        seq_one_proc_r,
        &seq_two_proc,
        match_score,
        mismatch_score,
        indel_score,
        true,
    );

    // Add left score to reversed right score and locate pivot
    let mut best_score: f64 = f64::MIN;
    let mut pivot: usize = 0;
    for idx in 0..=seq_two_proc_len {
        let combined_score = score_l[idx] + score_r[seq_two_proc_len - idx];
        if combined_score > best_score {
            best_score = combined_score;
            pivot = idx;
        }
    }

    // Divide and conquer!
    let seq_one_proc_pivot_l = seq_one_proc[0..seq_one_proc_mid].to_vec();
    let seq_one_proc_pivot_r = seq_one_proc[seq_one_proc_mid..seq_one_proc_len].to_vec();

    let seq_two_proc_pivot_l = seq_two_proc[0..pivot].to_vec();
    let seq_two_proc_pivot_r = seq_two_proc[pivot..seq_two_proc_len].to_vec();

    let (aligned_seq_one_proc_pivot_l, aligned_seq_two_proc_pivot_l) = hirschberg(
        seq_one_proc_pivot_l,
        seq_two_proc_pivot_l,
        match_score,
        mismatch_score,
        indel_score,
        gap_val,
    )
    .expect("Expected Hirschberg's algorithm to run successfully for left subsequence");
    let (mut aligned_seq_one_proc_pivot_r, mut aligned_seq_two_proc_pivot_r) = hirschberg(
        seq_one_proc_pivot_r,
        seq_two_proc_pivot_r,
        match_score,
        mismatch_score,
        indel_score,
        gap_val,
    )
    .expect("Expected Hirschberg's algorithm to run successfully for right subsequence");

    // Combine together, swapping order again if we did so at the beginning.
    let mut aligned_seq_one_proc = aligned_seq_one_proc_pivot_l;
    aligned_seq_one_proc.append(&mut aligned_seq_one_proc_pivot_r);
    let mut aligned_seq_two_proc = aligned_seq_two_proc_pivot_l;
    aligned_seq_two_proc.append(&mut aligned_seq_two_proc_pivot_r);

    let (aligned_seq_one, aligned_seq_two) = if swapped {
        (aligned_seq_two_proc, aligned_seq_one_proc)
    } else {
        (aligned_seq_one_proc, aligned_seq_two_proc)
    };
    Ok((aligned_seq_one, aligned_seq_two))
}

/// A Python module implemented in Rust.
#[pymodule]
fn _sequence_align(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(needleman_wunsch, m)?)?;
    m.add_function(wrap_pyfunction!(hirschberg, m)?)?;
    Ok(())
}

// Assuming _GAP_VAL is defined elsewhere as a constant.
// For example: const _GAP_VAL: usize = usize::MAX;
// Adjust according to your actual gap value and usage context.

/// Maps entries of two sequences to unique indices, excluding a specific gap symbol.
///
/// # Arguments
///
/// * `seq_a` - A sequence of symbols.
/// * `seq_b` - Another sequence of symbols.
/// * `gap` - The gap symbol that should not appear in either sequence.
///
/// # Returns
///
/// A tuple containing a vector of unique symbols, and two vectors representing the
/// indices of the symbols in `seq_a` and `seq_b`, respectively.
///
/// # Errors
///
/// Returns an error if the gap symbol is found in either of the sequences.
fn entry2idx(
    seq_a: &[&str],
    seq_b: &[&str],
    gap: &str,
) -> anyhow::Result<(Vec<String>, Vec<i64>, Vec<i64>)> {
    let mut unique_symbols: HashSet<String> = seq_a.iter().map(|s| s.to_string()).collect();
    unique_symbols.extend(seq_b.iter().map(|s| s.to_string()));

    if unique_symbols.contains(gap) {
        return Err(anyhow::anyhow!("Gap entry {:?} found in seq_a and/or seq_b; must not exist in either", gap));
    }

    let idx2symbol: Vec<String> = unique_symbols.into_iter().collect();
    let symbol2idx: std::collections::HashMap<String, i64> = idx2symbol
        .iter()
        .cloned()
        .enumerate()
        .map(|(idx, symbol)| (symbol, idx as i64))
        .collect();

    let seq_a_indices = seq_a.iter().map(|symbol| *symbol2idx.get(&symbol.to_string()).unwrap()).collect();
    let seq_b_indices = seq_b.iter().map(|symbol| *symbol2idx.get(&symbol.to_string()).unwrap()).collect();

    Ok((idx2symbol, seq_a_indices, seq_b_indices))
}

/// Converts arrays of indices back into sequences of symbols.
///
/// # Arguments
///
/// * `idx2symbol` - A vector mapping indices to symbols.
/// * `seq_a_indices_aligned` - Indices representing an aligned sequence.
/// * `seq_b_indices_aligned` - Indices representing another aligned sequence.
/// * `gap` - The gap symbol.
/// * `_gap_val` - The special value representing a gap in the sequences.
///
/// # Returns
///
/// A tuple containing two vectors representing the sequences reconstructed from the indices.
fn idx2entry(
    idx2symbol: &[String],
    seq_a_indices_aligned: &[i64],
    seq_b_indices_aligned: &[i64],
    gap: &str,
    gap_val: i64,
) -> (Vec<String>, Vec<String>) {
    let seq_a_aligned: Vec<String> = seq_a_indices_aligned
        .iter()
        .map(|&idx| if idx == gap_val { gap.to_string() } else { idx2symbol[idx as usize].to_string() })
        .collect();
    let seq_b_aligned: Vec<String> = seq_b_indices_aligned
        .iter()
        .map(|&idx| if idx == gap_val { gap.to_string() } else { idx2symbol[idx as usize].to_string() })
        .collect();

    (seq_a_aligned, seq_b_aligned)
}

