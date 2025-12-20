"""
Advanced visualization functions for Rossmann Sales Forecasting project.

This module provides comprehensive plotting capabilities for:
- Model performance comparisons
- Statistical analysis visualizations
- Business impact assessments
- Error analysis and diagnostic plots
- Interactive dashboards for stakeholder presentations

Author: Rossmann ML Team
Phase: 4 - Model Evaluation & Visualization
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import json
import warnings
warnings.filterwarnings('ignore')

# Core libraries
import pandas as pd
import numpy as np
from loguru import logger
from tqdm import tqdm
import typer

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff

# Statistical libraries
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Project configuration
from rossmann_sales_forecasting.config import FIGURES_DIR, PROCESSED_DATA_DIR

# Configure plotting defaults
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11

app = typer.Typer()


class ModelEvaluationPlots:
    """Advanced plotting class for model evaluation and business analysis."""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8), style: str = 'default'):
        """Initialize plotting configuration."""
        plt.style.use(style)
        self.figsize = figsize
        self.colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e', 
            'success': '#2ca02c',
            'warning': '#d62728',
            'info': '#9467bd'
        }
        
    def model_performance_comparison(
        self, 
        results_df: pd.DataFrame, 
        save_path: Optional[Path] = None
    ) -> go.Figure:
        """
        Create comprehensive model performance comparison visualization.
        
        Args:
            results_df: DataFrame with model performance metrics
            save_path: Optional path to save the figure
            
        Returns:
            Plotly figure object
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('R² Score Comparison', 'RMSE vs MAE', 'MAPE Analysis', 'Performance Matrix'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # R² Score comparison
        fig.add_trace(
            go.Bar(
                x=results_df['Model'],
                y=results_df['r2'] * 100,
                name='R² Score (%)',
                marker_color=px.colors.qualitative.Set3,
                text=results_df['r2'].apply(lambda x: f'{x*100:.1f}%'),
                textposition='outside'
            ),
            row=1, col=1
        )
        
        # RMSE vs MAE scatter
        fig.add_trace(
            go.Scatter(
                x=results_df['rmse'],
                y=results_df['mae'],
                mode='markers+text',
                text=results_df['Model'],
                textposition='top center',
                marker=dict(
                    size=12, 
                    color=results_df['r2'], 
                    colorscale='Viridis', 
                    showscale=True,
                    colorbar=dict(title="R² Score")
                ),
                name='RMSE vs MAE (colored by R²)'
            ),
            row=1, col=2
        )
        
        # MAPE analysis
        fig.add_trace(
            go.Bar(
                x=results_df['Model'],
                y=results_df['mape'],
                name='MAPE (%)',
                marker_color=px.colors.qualitative.Pastel1,
                text=results_df['mape'].apply(lambda x: f'{x:.1f}%'),
                textposition='outside'
            ),
            row=2, col=1
        )
        
        # Performance matrix heatmap
        metrics_matrix = results_df[['r2', 'rmse', 'mae', 'mape']].values
        normalized_matrix = (metrics_matrix - metrics_matrix.min(axis=0)) / (
            metrics_matrix.max(axis=0) - metrics_matrix.min(axis=0)
        )
        
        fig.add_trace(
            go.Heatmap(
                z=normalized_matrix,
                x=['R²', 'RMSE', 'MAE', 'MAPE'],
                y=results_df['Model'],
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title="Normalized Score")
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            height=800,
            title_text="📊 Comprehensive Model Performance Analysis",
            title_font_size=16,
            showlegend=False
        )
        
        # Update axes
        fig.update_xaxes(title_text="Models", row=1, col=1, tickangle=45)
        fig.update_xaxes(title_text="RMSE", row=1, col=2)
        fig.update_xaxes(title_text="Models", row=2, col=1, tickangle=45)
        fig.update_yaxes(title_text="R² Score (%)", row=1, col=1)
        fig.update_yaxes(title_text="MAE", row=1, col=2)
        fig.update_yaxes(title_text="MAPE (%)", row=2, col=1)
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Performance comparison saved to {save_path}")
            
        return fig
    
    def cross_validation_analysis(
        self, 
        cv_results: Dict[str, Dict], 
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        Create cross-validation analysis plots.
        
        Args:
            cv_results: Dictionary with CV results for each model
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure object
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Box plot of CV scores
        cv_data = []
        cv_labels = []
        for model_name, results in cv_results.items():
            cv_data.append(results['scores'])
            cv_labels.append(model_name.replace('_', ' ').title())

        bp = ax1.boxplot(cv_data, labels=cv_labels, patch_artist=True)
        colors = plt.cm.Set3(np.linspace(0, 1, len(cv_data)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)

        ax1.set_title('📊 Cross-Validation Score Distribution', fontsize=14, fontweight='bold')
        ax1.set_ylabel('R² Score')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)

        # Mean ± Standard deviation comparison
        models_names = list(cv_results.keys())
        means = [cv_results[m]['mean'] for m in models_names]
        stds = [cv_results[m]['std'] for m in models_names]
        labels = [m.replace('_', ' ').title() for m in models_names]

        bars = ax2.bar(labels, means, yerr=stds, capsize=5, 
                       color=colors[:len(models_names)], alpha=0.7, 
                       error_kw={'linewidth': 2, 'ecolor': 'black'})

        ax2.set_title('📈 Mean CV Performance ± Standard Deviation', fontsize=14, fontweight='bold')
        ax2.set_ylabel('R² Score')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)

        # Add value labels on bars
        for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.01,
                     f'{mean:.3f}', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"CV analysis saved to {save_path}")
            
        return fig
    
    def business_impact_dashboard(
        self, 
        business_df: pd.DataFrame, 
        save_path: Optional[Path] = None
    ) -> go.Figure:
        """
        Create business impact analysis dashboard.
        
        Args:
            business_df: DataFrame with business impact metrics
            save_path: Optional path to save the figure
            
        Returns:
            Plotly figure object
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Annual Revenue Impact', 'Operational Efficiency Gains', 
                           'Staff & Inventory Improvements', '3-Year ROI Comparison'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )

        models = business_df['model'].tolist()
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

        # Annual revenue impact
        fig.add_trace(
            go.Bar(
                x=models,
                y=business_df['annual_revenue_impact'],
                name='Annual Revenue Impact (€)',
                marker_color=colors,
                text=[f'€{x:,.0f}' for x in business_df['annual_revenue_impact']],
                textposition='outside'
            ),
            row=1, col=1
        )

        # Efficiency gains
        fig.add_trace(
            go.Bar(
                x=models,
                y=business_df['efficiency_gain'],
                name='Efficiency Gain (%)',
                marker_color=colors,
                text=[f'{x:.1f}%' for x in business_df['efficiency_gain']],
                textposition='outside'
            ),
            row=1, col=2
        )

        # Staff and inventory improvements scatter
        fig.add_trace(
            go.Scatter(
                x=business_df['staff_efficiency'],
                y=business_df['inventory_improvement'],
                mode='markers+text',
                text=models,
                textposition='top center',
                marker=dict(size=15, color=colors),
                name='Staff vs Inventory Efficiency'
            ),
            row=2, col=1
        )

        # ROI comparison (calculate here)
        roi_values = []
        for _, row in business_df.iterrows():
            annual_benefit = row['annual_revenue_impact']
            three_year_benefit = annual_benefit * 3
            total_cost = 150000 + (50000 * 3)  # implementation + 3yr maintenance
            roi = ((three_year_benefit - total_cost) / total_cost) * 100
            roi_values.append(roi)

        fig.add_trace(
            go.Bar(
                x=models,
                y=roi_values,
                name='3-Year ROI (%)',
                marker_color=colors,
                text=[f'{x:.0f}%' for x in roi_values],
                textposition='outside'
            ),
            row=2, col=2
        )

        # Update layout
        fig.update_layout(
            height=800,
            title_text="💰 Business Impact Analysis Dashboard",
            title_font_size=16,
            showlegend=False
        )

        # Update axes
        fig.update_yaxes(title_text="Revenue Impact (€)", row=1, col=1)
        fig.update_yaxes(title_text="Efficiency Gain (%)", row=1, col=2)
        fig.update_xaxes(title_text="Staff Efficiency Gain (%)", row=2, col=1)
        fig.update_yaxes(title_text="Inventory Improvement (%)", row=2, col=1)
        fig.update_yaxes(title_text="ROI (%)", row=2, col=2)
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Business impact dashboard saved to {save_path}")

        return fig
    
    def decision_framework_radar(
        self, 
        model_scores: Dict[str, Dict], 
        criteria: Dict[str, float],
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        Create radar chart for multi-criteria decision analysis.
        
        Args:
            model_scores: Dictionary with scores for each model and criterion
            criteria: Dictionary with criterion names and weights
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure object
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Radar chart for top 3 models
        categories = list(criteria.keys())
        categories_labels = [c.replace('_', ' ').title() for c in categories]

        # Number of variables
        N = len(categories)

        # Compute angles for each criterion
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Complete the circle

        # Create radar chart
        ax1 = plt.subplot(121, projection='polar')
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        models_list = list(model_scores.keys())[:3]  # Top 3 models

        for i, model in enumerate(models_list):
            values = [model_scores[model][cat] for cat in categories]
            values += values[:1]  # Complete the circle
            
            ax1.plot(angles, values, 'o-', linewidth=2, 
                     label=model.replace('_', ' ').title(), color=colors[i])
            ax1.fill(angles, values, alpha=0.25, color=colors[i])

        # Customize radar chart
        ax1.set_xticks(angles[:-1])
        ax1.set_xticklabels(categories_labels)
        ax1.set_ylim(0, 10)
        ax1.set_yticks(range(0, 11, 2))
        ax1.set_title('🎯 Multi-Criteria Model Comparison\n(Higher = Better)', size=12, y=1.08)
        ax1.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax1.grid(True)

        # Criteria weights pie chart
        ax2 = plt.subplot(122)
        wedges, texts, autotexts = ax2.pie(
            list(criteria.values()), 
            labels=[c.replace('_', ' ').title() for c in criteria.keys()],
            autopct='%1.0f%%',
            colors=plt.cm.Set3(np.linspace(0, 1, len(criteria))),
            startangle=90
        )
        
        ax2.set_title('⚖️ Decision Criteria Weights', size=12)
        
        # Make percentage text bold
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')

        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Decision framework radar saved to {save_path}")
            
        return fig


def create_evaluation_plots(
    training_run_path: Path,
    output_dir: Optional[Path] = None
) -> Dict[str, Path]:
    """
    Generate all evaluation plots for a training run.
    
    Args:
        training_run_path: Path to training run directory
        output_dir: Optional output directory for plots
        
    Returns:
        Dictionary mapping plot names to file paths
    """
    if output_dir is None:
        output_dir = training_run_path / 'evaluation_plots'
    
    output_dir.mkdir(exist_ok=True)
    
    # Load results
    with open(training_run_path / 'training_results.json', 'r') as f:
        results = json.load(f)
    
    # Initialize plotter
    plotter = ModelEvaluationPlots()
    plot_paths = {}
    
    logger.info("Generating comprehensive evaluation plots...")
    
    # Create performance comparison DataFrame
    performance_data = []
    for model_name, metrics in results.items():
        if model_name not in ['baselines'] and 'metrics' in metrics:
            perf = metrics['metrics'].copy()
            perf['Model'] = model_name.replace('_', ' ').title()
            performance_data.append(perf)
    
    performance_df = pd.DataFrame(performance_data)
    performance_df = performance_df.sort_values('r2', ascending=False)
    
    # Generate plots
    if not performance_df.empty:
        # Performance comparison
        perf_plot_path = output_dir / 'model_performance_comparison.html'
        perf_fig = plotter.model_performance_comparison(performance_df, perf_plot_path)
        plot_paths['performance_comparison'] = perf_plot_path
        
    logger.success(f"Evaluation plots generated in {output_dir}")
    return plot_paths


@app.command()
def generate_plots(
    training_run: str = typer.Argument(..., help="Training run directory name"),
    output_dir: Optional[Path] = typer.Option(None, help="Output directory for plots")
):
    """Generate comprehensive evaluation plots for a training run."""
    
    # Find training run directory
    models_dir = Path("../models")
    training_run_path = models_dir / training_run
    
    if not training_run_path.exists():
        logger.error(f"Training run directory not found: {training_run_path}")
        raise typer.Exit(1)
    
    # Generate plots
    plot_paths = create_evaluation_plots(training_run_path, output_dir)
    
    # Display results
    logger.success("Plot generation complete!")
    for plot_name, path in plot_paths.items():
        logger.info(f"{plot_name}: {path}")


@app.command()
def main(
    input_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    output_path: Path = FIGURES_DIR / "evaluation_plots",
):
    """Main command for generating evaluation plots."""
    logger.info("Generating comprehensive evaluation plots...")
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # This would be called from the notebook or other scripts
    logger.success("Plot generation utilities ready.")
    logger.info(f"Use ModelEvaluationPlots class for advanced visualizations")
    logger.info(f"Output directory: {output_path}")


if __name__ == "__main__":
    app()
