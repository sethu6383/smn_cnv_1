#!/usr/bin/env python3

"""
dual_segmentation.py - Dual segmentation approach using CBS and HMM for CNV detection
Usage: python dual_segmentation.py <z_scores_file> <output_file> [--method both] [--min-segment-size 3]
"""

import sys
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# Try to import specialized packages, fall back to custom implementations if not available
try:
    from hmmlearn import hmm
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    print("Warning: hmmlearn not available, using custom HMM implementation")

class CircularBinarySegmentation:
    """Circular Binary Segmentation for change-point detection in CNV data."""
    
    def __init__(self, alpha=0.01, min_segment_size=3, max_segments=50):
        self.alpha = alpha
        self.min_segment_size = min_segment_size
        self.max_segments = max_segments
    
    def _t_statistic(self, data, start, split, end):
        """Calculate t-statistic for potential change-point."""
        if split - start < self.min_segment_size or end - split < self.min_segment_size:
            return 0, 1
        
        left_data = data[start:split]
        right_data = data[split:end]
        
        if len(left_data) == 0 or len(right_data) == 0:
            return 0, 1
        
        # Remove NaN values
        left_data = left_data[~np.isnan(left_data)]
        right_data = right_data[~np.isnan(right_data)]
        
        if len(left_data) < 2 or len(right_data) < 2:
            return 0, 1
        
        # Two-sample t-test
        t_stat, p_value = stats.ttest_ind(left_data, right_data, equal_var=False)
        return abs(t_stat), p_value
    
    def _find_best_changepoint(self, data, start, end):
        """Find the best change-point in a segment."""
        best_t_stat = 0
        best_split = None
        best_p_value = 1
        
        min_split = start + self.min_segment_size
        max_split = end - self.min_segment_size
        
        for split in range(min_split, max_split + 1):
            t_stat, p_value = self._t_statistic(data, start, split, end)
            
            if t_stat > best_t_stat and p_value < self.alpha:
                best_t_stat = t_stat
                best_split = split
                best_p_value = p_value
        
        return best_split, best_t_stat, best_p_value
    
    def segment(self, data, positions=None):
        """Perform CBS segmentation on the data."""
        if positions is None:
            positions = np.arange(len(data))
        
        # Initialize segments list
        segments = [(0, len(data))]
        change_points = []
        
        iteration = 0
        while iteration < self.max_segments:
            best_segment = None
            best_split = None
            best_t_stat = 0
            best_p_value = 1
            
            # Find best change-point across all current segments
            for i, (start, end) in enumerate(segments):
                if end - start < 2 * self.min_segment_size:
                    continue
                
                split, t_stat, p_value = self._find_best_changepoint(data, start, end)
                
                if split is not None and t_stat > best_t_stat:
                    best_segment = i
                    best_split = split
                    best_t_stat = t_stat
                    best_p_value = p_value
            
            # If no significant change-point found, stop
            if best_split is None:
                break
            
            # Split the best segment
            start, end = segments[best_segment]
            segments[best_segment] = (start, best_split)
            segments.insert(best_segment + 1, (best_split, end))
            change_points.append(best_split)
            
            iteration += 1
        
        # Calculate segment statistics
        segment_stats = []
        for start, end in segments:
            segment_data = data[start:end]
            segment_data = segment_data[~np.isnan(segment_data)]
            
            if len(segment_data) > 0:
                segment_stats.append({
                    'start': start,
                    'end': end,
                    'start_pos': positions[start] if start < len(positions) else start,
                    'end_pos': positions[end-1] if end-1 < len(positions) else end-1,
                    'length': end - start,
                    'mean': np.mean(segment_data),
                    'std': np.std(segment_data) if len(segment_data) > 1 else 0,
                    'median': np.median(segment_data),
                    'n_points': len(segment_data)
                })
        
        return segment_stats, sorted(change_points)

class CustomHMM:
    """Custom Hidden Markov Model implementation for copy number states."""
    
    def __init__(self, n_states=5, max_iter=100, tol=1e-6):
        self.n_states = n_states
        self.max_iter = max_iter
        self.tol = tol
        self.state_means_ = None
        self.state_vars_ = None
        self.transmat_ = None
        self.startprob_ = None
    
    def _initialize_parameters(self, X):
        """Initialize HMM parameters."""
        n_obs = len(X)
        
        # Initialize state means using k-means
        valid_X = X[~np.isnan(X)]
        if len(valid_X) < self.n_states:
            # Fallback initialization
            self.state_means_ = np.linspace(np.nanmin(X), np.nanmax(X), self.n_states)
        else:
            kmeans = KMeans(n_clusters=self.n_states, random_state=42, n_init=10)
            kmeans.fit(valid_X.reshape(-1, 1))
            self.state_means_ = np.sort(kmeans.cluster_centers_.flatten())
        
        # Initialize variances
        self.state_vars_ = np.ones(self.n_states)
        
        # Initialize transition matrix (slightly favoring staying in same state)
        self.transmat_ = np.ones((self.n_states, self.n_states)) * 0.1
        np.fill_diagonal(self.transmat_, 0.7)
        self.transmat_ = self.transmat_ / self.transmat_.sum(axis=1, keepdims=True)
        
        # Initialize start probabilities (uniform)
        self.startprob_ = np.ones(self.n_states) / self.n_states
    
    def _emission_probability(self, obs, state):
        """Calculate emission probability for observation given state."""
        if np.isnan(obs):
            return 1.0  # Missing data - uniform probability
        
        mean = self.state_means_[state]
        var = max(self.state_vars_[state], 1e-6)  # Avoid division by zero
        
        # Gaussian emission probability
        prob = (1.0 / np.sqrt(2 * np.pi * var)) * np.exp(-0.5 * (obs - mean)**2 / var)
        return max(prob, 1e-10)  # Avoid log(0)
    
    def _forward_algorithm(self, X):
        """Forward algorithm for HMM."""
        n_obs = len(X)
        alpha = np.zeros((n_obs, self.n_states))
        
        # Initialize
        for state in range(self.n_states):
            alpha[0, state] = self.startprob_[state] * self._emission_probability(X[0], state)
        
        # Forward pass
        for t in range(1, n_obs):
            for state in range(self.n_states):
                alpha[t, state] = sum(
                    alpha[t-1, prev_state] * self.transmat_[prev_state, state]
                    for prev_state in range(self.n_states)
                ) * self._emission_probability(X[t], state)
        
        return alpha
    
    def _backward_algorithm(self, X):
        """Backward algorithm for HMM."""
        n_obs = len(X)
        beta = np.zeros((n_obs, self.n_states))
        
        # Initialize
        beta[n_obs-1, :] = 1.0
        
        # Backward pass
        for t in range(n_obs-2, -1, -1):
            for state in range(self.n_states):
                beta[t, state] = sum(
                    self.transmat_[state, next_state] * 
                    self._emission_probability(X[t+1], next_state) * 
                    beta[t+1, next_state]
                    for next_state in range(self.n_states)
                )
        
        return beta
    
    def _viterbi_algorithm(self, X):
        """Viterbi algorithm for finding most likely state sequence."""
        n_obs = len(X)
        delta = np.zeros((n_obs, self.n_states))
        psi = np.zeros((n_obs, self.n_states), dtype=int)
        
        # Initialize
        for state in range(self.n_states):
            delta[0, state] = np.log(self.startprob_[state]) + np.log(self._emission_probability(X[0], state))
        
        # Forward pass
        for t in range(1, n_obs):
            for state in range(self.n_states):
                transitions = [
                    delta[t-1, prev_state] + np.log(max(self.transmat_[prev_state, state], 1e-10))
                    for prev_state in range(self.n_states)
                ]
                psi[t, state] = np.argmax(transitions)
                delta[t, state] = max(transitions) + np.log(self._emission_probability(X[t], state))
        
        # Backward pass (traceback)
        states = np.zeros(n_obs, dtype=int)
        states[n_obs-1] = np.argmax(delta[n_obs-1, :])
        
        for t in range(n_obs-2, -1, -1):
            states[t] = psi[t+1, states[t+1]]
        
        return states
    
    def fit_predict(self, X):
        """Fit HMM and predict states."""
        self._initialize_parameters(X)
        
        for iteration in range(self.max_iter):
            # E-step: Forward-backward algorithm
            alpha = self._forward_algorithm(X)
            beta = self._backward_algorithm(X)
            
            # Calculate gamma (state posteriors)
            gamma = alpha * beta
            gamma = gamma / gamma.sum(axis=1, keepdims=True)
            
            # M-step: Update parameters
            old_means = self.state_means_.copy()
            
            for state in range(self.n_states):
                # Update means
                weights = gamma[:, state]
                valid_mask = ~np.isnan(X)
                if weights[valid_mask].sum() > 0:
                    self.state_means_[state] = np.average(X[valid_mask], weights=weights[valid_mask])
                
                # Update variances
                if weights[valid_mask].sum() > 1:
                    diff_sq = (X[valid_mask] - self.state_means_[state])**2
                    self.state_vars_[state] = np.average(diff_sq, weights=weights[valid_mask])
                    self.state_vars_[state] = max(self.state_vars_[state], 1e-6)
            
            # Check convergence
            if np.allclose(old_means, self.state_means_, atol=self.tol):
                break
        
        # Predict states using Viterbi algorithm
        states = self._viterbi_algorithm(X)
        return states

class HMMCNVAnalyzer:
    """HMM-based CNV analyzer with support for both hmmlearn and custom implementation."""
    
    def __init__(self, n_states=5, method='auto'):
        self.n_states = n_states
        self.method = method
        self.model = None
        self.state_to_cn_mapping = None
    
    def _map_states_to_copy_numbers(self, states, state_means):
        """Map HMM states to copy numbers."""
        # Sort states by their means
        sorted_states = np.argsort(state_means)
        
        # Map to copy numbers (0, 1, 2, 3, 4+)
        state_to_cn = {}
        n_states = len(sorted_states)
        
        if n_states == 5:
            # Standard 5-state model
            cn_mapping = [0, 1, 2, 3, 4]
        elif n_states == 3:
            # Simplified 3-state model
            cn_mapping = [1, 2, 3]
        else:
            # General case
            cn_mapping = list(range(max(0, 2 - n_states//2), 2 + (n_states - n_states//2)))
        
        for i, state in enumerate(sorted_states):
            if i < len(cn_mapping):
                state_to_cn[state] = cn_mapping[i]
            else:
                state_to_cn[state] = cn_mapping[-1]  # Cap at highest CN
        
        self.state_to_cn_mapping = state_to_cn
        
        # Convert state sequence to copy numbers
        copy_numbers = np.array([state_to_cn[state] for state in states])
        return copy_numbers
    
    def analyze(self, z_scores, positions=None):
        """Analyze copy numbers using HMM."""
        if positions is None:
            positions = np.arange(len(z_scores))
        
        # Handle missing data
        valid_mask = ~np.isnan(z_scores)
        if valid_mask.sum() == 0:
            return np.full(len(z_scores), 2), {}  # Return normal CN if all missing
        
        try:
            if HMM_AVAILABLE and self.method in ['auto', 'hmmlearn']:
                # Use hmmlearn implementation
                model = hmm.GaussianHMM(n_components=self.n_states, covariance_type="full", n_iter=100)
                
                # Prepare data for hmmlearn (requires 2D array)
                X_train = z_scores[valid_mask].reshape(-1, 1)
                model.fit(X_train)
                
                # Predict on full dataset (handling missing data)
                X_full = np.copy(z_scores)
                X_full[~valid_mask] = np.nanmean(z_scores)  # Fill missing with mean
                states = model.predict(X_full.reshape(-1, 1))
                
                state_means = model.means_.flatten()
                self.model = model
                
            else:
                # Use custom implementation
                custom_hmm = CustomHMM(n_states=self.n_states)
                states = custom_hmm.fit_predict(z_scores)
                state_means = custom_hmm.state_means_
                self.model = custom_hmm
            
            # Map states to copy numbers
            copy_numbers = self._map_states_to_copy_numbers(states, state_means)
            
            # Calculate segment statistics
            segments = []
            current_cn = copy_numbers[0]
            start_idx = 0
            
            for i in range(1, len(copy_numbers)):
                if copy_numbers[i] != current_cn:
                    # End current segment
                    segment_z_scores = z_scores[start_idx:i]
                    valid_z = segment_z_scores[~np.isnan(segment_z_scores)]
                    
                    segments.append({
                        'start': start_idx,
                        'end': i,
                        'start_pos': positions[start_idx],
                        'end_pos': positions[i-1],
                        'copy_number': current_cn,
                        'length': i - start_idx,
                        'mean_z_score': np.mean(valid_z) if len(valid_z) > 0 else np.nan,
                        'std_z_score': np.std(valid_z) if len(valid_z) > 1 else np.nan,
                        'n_points': len(valid_z)
                    })
                    
                    # Start new segment
                    current_cn = copy_numbers[i]
                    start_idx = i
            
            # Add final segment
            segment_z_scores = z_scores[start_idx:]
            valid_z = segment_z_scores[~np.isnan(segment_z_scores)]
            
            segments.append({
                'start': start_idx,
                'end': len(copy_numbers),
                'start_pos': positions[start_idx],
                'end_pos': positions[-1],
                'copy_number': current_cn,
                'length': len(copy_numbers) - start_idx,
                'mean_z_score': np.mean(valid_z) if len(valid_z) > 0 else np.nan,
                'std_z_score': np.std(valid_z) if len(valid_z) > 1 else np.nan,
                'n_points': len(valid_z)
            })
            
            analysis_info = {
                'method': 'hmmlearn' if HMM_AVAILABLE and self.method in ['auto', 'hmmlearn'] else 'custom',
                'n_states': self.n_states,
                'state_means': state_means,
                'state_to_cn_mapping': self.state_to_cn_mapping,
                'n_segments': len(segments)
            }
            
            return copy_numbers, segments, analysis_info
            
        except Exception as e:
            print(f"Warning: HMM analysis failed ({e}). Using fallback method.")
            # Fallback: simple threshold-based approach
            copy_numbers = np.full(len(z_scores), 2)  # Default to normal
            copy_numbers[z_scores <= -2.5] = 0
            copy_numbers[(z_scores > -2.5) & (z_scores <= -1.5)] = 1
            copy_numbers[(z_scores > 1.5) & (z_scores <= 2.5)] = 3
            copy_numbers[z_scores > 2.5] = 4
            
            return copy_numbers, [], {'method': 'fallback', 'error': str(e)}

def perform_dual_segmentation(z_scores_df, method='both', min_segment_size=3):
    """Perform dual segmentation analysis on Z-scores data."""
    results = []
    
    # Group by sample
    for sample_id in z_scores_df['sample_id'].unique():
        sample_data = z_scores_df[z_scores_df['sample_id'] == sample_id].copy()
        sample_data = sample_data.sort_values('exon')  # Ensure consistent ordering
        
        # Extract Z-scores and positions
        z_scores = sample_data['z_score'].values
        robust_z_scores = sample_data['robust_z_score'].values
        exon_names = sample_data['exon'].values
        positions = np.arange(len(z_scores))
        
        sample_result = {
            'sample_id': sample_id,
            'exon_names': exon_names,
            'z_scores': z_scores,
            'robust_z_scores': robust_z_scores
        }
        
        # CBS Analysis
        if method in ['cbs', 'both']:
            try:
                cbs = CircularBinarySegmentation(min_segment_size=min_segment_size)
                cbs_segments, cbs_changepoints = cbs.segment(robust_z_scores, positions)
                
                sample_result['cbs_segments'] = cbs_segments
                sample_result['cbs_changepoints'] = cbs_changepoints
                sample_result['cbs_success'] = True
            except Exception as e:
                print(f"Warning: CBS failed for sample {sample_id}: {e}")
                sample_result['cbs_segments'] = []
                sample_result['cbs_changepoints'] = []
                sample_result['cbs_success'] = False
        
        # HMM Analysis
        if method in ['hmm', 'both']:
            try:
                hmm_analyzer = HMMCNVAnalyzer(n_states=5)
                hmm_copy_numbers, hmm_segments, hmm_info = hmm_analyzer.analyze(robust_z_scores, positions)
                
                sample_result['hmm_copy_numbers'] = hmm_copy_numbers
                sample_result['hmm_segments'] = hmm_segments
                sample_result['hmm_info'] = hmm_info
                sample_result['hmm_success'] = True
            except Exception as e:
                print(f"Warning: HMM failed for sample {sample_id}: {e}")
                sample_result['hmm_copy_numbers'] = np.full(len(z_scores), 2)
                sample_result['hmm_segments'] = []
                sample_result['hmm_info'] = {'method': 'failed', 'error': str(e)}
                sample_result['hmm_success'] = False
        
        # Consensus analysis (if both methods used)
        if method == 'both' and sample_result.get('cbs_success', False) and sample_result.get('hmm_success', False):
            try:
                consensus_cn = consensus_copy_number_calls(
                    sample_result['cbs_segments'], 
                    sample_result['hmm_copy_numbers'],
                    len(z_scores)
                )
                sample_result['consensus_copy_numbers'] = consensus_cn
            except Exception as e:
                print(f"Warning: Consensus analysis failed for sample {sample_id}: {e}")
                sample_result['consensus_copy_numbers'] = sample_result['hmm_copy_numbers']
        
        results.append(sample_result)
    
    return results

def consensus_copy_number_calls(cbs_segments, hmm_copy_numbers, n_positions):
    """Generate consensus copy number calls from CBS and HMM results."""
    consensus = np.full(n_positions, 2, dtype=int)  # Default to normal
    
    # Create CBS-based copy number array
    cbs_copy_numbers = np.full(n_positions, 2, dtype=int)
    for segment in cbs_segments:
        start, end = segment['start'], segment['end']
        mean_z = segment['mean']
        
        # Convert mean Z-score to copy number estimate
        if mean_z <= -2.5:
            cn = 0
        elif mean_z <= -1.5:
            cn = 1
        elif mean_z <= 1.5:
            cn = 2
        elif mean_z <= 2.5:
            cn = 3
        else:
            cn = 4
        
        cbs_copy_numbers[start:end] = cn
    
    # Consensus logic: agree if both methods give same result, otherwise use confidence-based decision
    for i in range(n_positions):
        cbs_cn = cbs_copy_numbers[i]
        hmm_cn = hmm_copy_numbers[i]
        
        if cbs_cn == hmm_cn:
            consensus[i] = cbs_cn
        else:
            # Conflict resolution: prefer HMM for intermediate states, CBS for extreme states
            if min(cbs_cn, hmm_cn) <= 1 or max(cbs_cn, hmm_cn) >= 3:
                # Extreme copy numbers: prefer CBS (better at detecting sharp changes)
                consensus[i] = cbs_cn
            else:
                # Intermediate copy numbers: prefer HMM (better at handling noise)
                consensus[i] = hmm_cn
    
    return consensus

def create_dual_segmentation_plots(segmentation_results, output_dir):
    """Create visualization plots for dual segmentation results."""
    plot_dir = Path(output_dir) / 'plots'
    plot_dir.mkdir(exist_ok=True)
    
    # Plot individual samples
    for result in segmentation_results[:6]:  # Limit to first 6 samples for readability
        sample_id = result['sample_id']
        
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        fig.suptitle(f'Dual Segmentation Analysis - {sample_id}', fontsize=16, fontweight='bold')
        
        x_pos = np.arange(len(result['exon_names']))
        z_scores = result['z_scores']
        robust_z_scores = result['robust_z_scores']
        
        # Plot 1: Raw data with CBS segments
        ax1 = axes[0]
        ax1.plot(x_pos, z_scores, 'o-', alpha=0.7, label='Standard Z-scores')
        ax1.plot(x_pos, robust_z_scores, 's-', alpha=0.7, label='Robust Z-scores')
        
        if 'cbs_segments' in result:
            for segment in result['cbs_segments']:
                start, end = segment['start'], segment['end']
                mean_val = segment['mean']
                ax1.axhspan(mean_val - 0.1, mean_val + 0.1, 
                           xmin=start/len(x_pos), xmax=end/len(x_pos), 
                           alpha=0.3, color='red', label='CBS Segments' if segment == result['cbs_segments'][0] else '')
                ax1.axvline(x=start, color='red', linestyle='--', alpha=0.5)
        
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax1.axhline(y=-1.5, color='orange', linestyle='--', alpha=0.5)
        ax1.axhline(y=1.5, color='orange', linestyle='--', alpha=0.5)
        ax1.set_title('CBS Segmentation')
        ax1.set_ylabel('Z-score')
        ax1.legend()
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(result['exon_names'], rotation=45)
        
        # Plot 2: HMM copy number states
        ax2 = axes[1]
        ax2.plot(x_pos, robust_z_scores, 'o-', alpha=0.5, color='gray', label='Robust Z-scores')
        
        if 'hmm_copy_numbers' in result:
            hmm_cn = result['hmm_copy_numbers']
            colors = ['red', 'orange', 'green', 'blue', 'purple']
            for i, cn in enumerate(hmm_cn):
                color = colors[min(cn, 4)] if cn < len(colors) else 'black'
                ax2.scatter(x_pos[i], robust_z_scores[i], c=color, s=100, alpha=0.7,
                           label=f'CN={cn}' if cn not in [hmm_cn[j] for j in range(i)] else '')
        
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.set_title('HMM Copy Number States')
        ax2.set_ylabel('Z-score')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(result['exon_names'], rotation=45)
        
        # Plot 3: Consensus (if available)
        ax3 = axes[2]
        ax3.plot(x_pos, robust_z_scores, 'o-', alpha=0.5, color='gray', label='Robust Z-scores')
        
        if 'consensus_copy_numbers' in result:
            consensus_cn = result['consensus_copy_numbers']
            colors = ['red', 'orange', 'green', 'blue', 'purple']
            for i, cn in enumerate(consensus_cn):
                color = colors[min(cn, 4)] if cn < len(colors) else 'black'
                ax3.scatter(x_pos[i], robust_z_scores[i], c=color, s=100, alpha=0.7, marker='D',
                           label=f'Consensus CN={cn}' if cn not in [consensus_cn[j] for j in range(i)] else '')
        elif 'hmm_copy_numbers' in result:
            # Fallback to HMM if consensus not available
            hmm_cn = result['hmm_copy_numbers']
            colors = ['red', 'orange', 'green', 'blue', 'purple']
            for i, cn in enumerate(hmm_cn):
                color = colors[min(cn, 4)] if cn < len(colors) else 'black'
                ax3.scatter(x_pos[i], robust_z_scores[i], c=color, s=100, alpha=0.7, marker='D',
                           label=f'Final CN={cn}' if cn not in [hmm_cn[j] for j in range(i)] else '')
        
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax3.set_title('Consensus/Final Copy Numbers')
        ax3.set_ylabel('Z-score')
        ax3.set_xlabel('Exon')
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(result['exon_names'], rotation=45)
        
        plt.tight_layout()
        plt.savefig(plot_dir / f'{sample_id}_dual_segmentation.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Summary plot: Method comparison
    plt.figure(figsize=(12, 8))
    
    method_comparison = {'CBS': 0, 'HMM': 0, 'Both': 0, 'Failed': 0}
    for result in segmentation_results:
        cbs_success = result.get('cbs_success', False)
        hmm_success = result.get('hmm_success', False)
        
        if cbs_success and hmm_success:
            method_comparison['Both'] += 1
        elif cbs_success:
            method_comparison['CBS'] += 1
        elif hmm_success:
            method_comparison['HMM'] += 1
        else:
            method_comparison['Failed'] += 1
    
    plt.pie(method_comparison.values(), labels=method_comparison.keys(), 
            autopct='%1.1f%%', startangle=90)
    plt.title('Segmentation Method Success Rate')
    plt.savefig(plot_dir / 'method_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Dual segmentation plots saved to: {plot_dir}")

def save_segmentation_results(segmentation_results, output_file):
    """Save segmentation results to files."""
    output_dir = Path(output_file).parent
    
    # Prepare data for main results file
    results_data = []
    
    for result in segmentation_results:
        sample_id = result['sample_id']
        exon_names = result['exon_names']
        z_scores = result['z_scores']
        robust_z_scores = result['robust_z_scores']
        
        # Get copy numbers (prefer consensus, fallback to HMM, then to default)
        if 'consensus_copy_numbers' in result:
            copy_numbers = result['consensus_copy_numbers']
            method = 'consensus'
        elif 'hmm_copy_numbers' in result:
            copy_numbers = result['hmm_copy_numbers']
            method = 'hmm'
        else:
            copy_numbers = np.full(len(exon_names), 2)  # Default to normal
            method = 'default'
        
        # Create individual records
        for i in range(len(exon_names)):
            results_data.append({
                'sample_id': sample_id,
                'exon': exon_names[i],
                'z_score': z_scores[i],
                'robust_z_score': robust_z_scores[i],
                'copy_number': copy_numbers[i],
                'segmentation_method': method,
                'cbs_available': result.get('cbs_success', False),
                'hmm_available': result.get('hmm_success', False)
            })
    
    # Save main results
    results_df = pd.DataFrame(results_data)
    results_df.to_csv(output_file, index=False, sep='\t')
    
    # Save detailed segmentation info
    segments_data = []
    for result in segmentation_results:
        sample_id = result['sample_id']
        
        # CBS segments
        if 'cbs_segments' in result:
            for i, segment in enumerate(result['cbs_segments']):
                segments_data.append({
                    'sample_id': sample_id,
                    'method': 'CBS',
                    'segment_id': i,
                    'start_index': segment['start'],
                    'end_index': segment['end'],
                    'start_position': segment['start_pos'],
                    'end_position': segment['end_pos'],
                    'length': segment['length'],
                    'mean_value': segment['mean'],
                    'std_value': segment['std'],
                    'median_value': segment['median'],
                    'n_points': segment['n_points']
                })
        
        # HMM segments
        if 'hmm_segments' in result:
            for i, segment in enumerate(result['hmm_segments']):
                segments_data.append({
                    'sample_id': sample_id,
                    'method': 'HMM',
                    'segment_id': i,
                    'start_index': segment['start'],
                    'end_index': segment['end'],
                    'start_position': segment['start_pos'],
                    'end_position': segment['end_pos'],
                    'length': segment['length'],
                    'copy_number': segment['copy_number'],
                    'mean_z_score': segment['mean_z_score'],
                    'std_z_score': segment['std_z_score'],
                    'n_points': segment['n_points']
                })
    
    # Save segments data
    if segments_data:
        segments_df = pd.DataFrame(segments_data)
        segments_file = output_file.replace('.txt', '_segments.txt')
        segments_df.to_csv(segments_file, index=False, sep='\t')
        
        print(f"Segmentation results saved to: {output_file}")
        print(f"Detailed segments saved to: {segments_file}")
    
    return results_df

def main():
    parser = argparse.ArgumentParser(description='Dual segmentation CNV analysis using CBS and HMM')
    parser.add_argument('z_scores_file', help='Enhanced Z-scores file from control gene normalization')
    parser.add_argument('output_file', help='Output file for segmentation results')
    parser.add_argument('--method', choices=['cbs', 'hmm', 'both'], default='both',
                       help='Segmentation method to use (default: both)')
    parser.add_argument('--min-segment-size', type=int, default=3,
                       help='Minimum segment size for CBS (default: 3)')
    parser.add_argument('--n-states', type=int, default=5,
                       help='Number of states for HMM (default: 5)')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip creating plots')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_file).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading enhanced Z-scores data...")
    try:
        z_scores_df = pd.read_csv(args.z_scores_file, sep='\t')
        print(f"Loaded Z-scores for {len(z_scores_df)} sample-exon combinations")
        print(f"Samples: {len(z_scores_df['sample_id'].unique())}")
    except Exception as e:
        print(f"Error reading Z-scores file: {e}")
        sys.exit(1)
    
    print(f"Performing dual segmentation analysis using method: {args.method}")
    
    # Perform segmentation analysis
    segmentation_results = perform_dual_segmentation(
        z_scores_df, 
        method=args.method, 
        min_segment_size=args.min_segment_size
    )
    
    # Save results
    results_df = save_segmentation_results(segmentation_results, args.output_file)
    
    # Create plots
    if not args.no_plots:
        try:
            create_dual_segmentation_plots(segmentation_results, output_dir)
        except Exception as e:
            print(f"Warning: Could not create plots: {e}")
    
    # Print summary
    print(f"\nDual segmentation analysis completed!")
    print(f"Results saved to: {args.output_file}")
    
    # Analysis summary
    total_samples = len(segmentation_results)
    cbs_success = sum(1 for r in segmentation_results if r.get('cbs_success', False))
    hmm_success = sum(1 for r in segmentation_results if r.get('hmm_success', False))
    both_success = sum(1 for r in segmentation_results if r.get('cbs_success', False) and r.get('hmm_success', False))
    
    print(f"\nAnalysis Summary:")
    print(f"Total samples: {total_samples}")
    print(f"CBS successful: {cbs_success} ({100*cbs_success/total_samples:.1f}%)")
    print(f"HMM successful: {hmm_success} ({100*hmm_success/total_samples:.1f}%)")
    print(f"Both methods successful: {both_success} ({100*both_success/total_samples:.1f}%)")
    
    # Copy number distribution
    if not results_df.empty:
        print(f"\nCopy number distribution:")
        cn_dist = results_df['copy_number'].value_counts().sort_index()
        for cn, count in cn_dist.items():
            print(f"  CN={cn}: {count} calls")
        
        # Method distribution
        print(f"\nSegmentation method distribution:")
        method_dist = results_df['segmentation_method'].value_counts()
        for method, count in method_dist.items():
            print(f"  {method}: {count} calls")

if __name__ == "__main__":
    main()
