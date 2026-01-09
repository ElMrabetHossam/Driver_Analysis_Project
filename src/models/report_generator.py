"""
Driver Safety Report Generator

Generates PDF reports with:
- Driver safety score and grade
- Contract recommendation (Bonus/Retain/Probation/Terminate)
- Risk factors and recommendations
- Trip statistics and visualizations

Usage:
    python -m src.models.report_generator --input data/processed/training_data.csv --output reports/
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for PDF
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Import scorer
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.models.scorer import DriverScorer, ScoreBreakdown


# Contract Decision Thresholds (from original requirements)
CONTRACT_THRESHOLDS = {
    'bonus': 85,        # Score > 85: Bonus eligible
    'retain': 70,       # Score 70-85: Retain contract
    'probation': 50,    # Score 50-70: Mandatory training / Probation
    # Score < 50: Terminate contract
}


class ContractDecision:
    """Contract decision based on driver performance."""
    
    BONUS = "BONUS_ELIGIBLE"
    RETAIN = "RETAIN_CONTRACT"
    PROBATION = "MANDATORY_TRAINING"
    TERMINATE = "TERMINATE_CONTRACT"
    
    @staticmethod
    def get_decision(score: float) -> str:
        """Get contract decision based on score."""
        if score >= CONTRACT_THRESHOLDS['bonus']:
            return ContractDecision.BONUS
        elif score >= CONTRACT_THRESHOLDS['retain']:
            return ContractDecision.RETAIN
        elif score >= CONTRACT_THRESHOLDS['probation']:
            return ContractDecision.PROBATION
        else:
            return ContractDecision.TERMINATE
    
    @staticmethod
    def get_decision_color(decision: str) -> str:
        """Get color for decision."""
        colors = {
            ContractDecision.BONUS: '#00C853',      # Green
            ContractDecision.RETAIN: '#2196F3',     # Blue
            ContractDecision.PROBATION: '#FF9100',  # Orange
            ContractDecision.TERMINATE: '#FF1744',  # Red
        }
        return colors.get(decision, '#666666')
    
    @staticmethod
    def get_decision_description(decision: str) -> str:
        """Get human-readable description."""
        descriptions = {
            ContractDecision.BONUS: "Driver eligible for performance bonus. Excellent safety record.",
            ContractDecision.RETAIN: "Contract renewed. Satisfactory driving performance.",
            ContractDecision.PROBATION: "Mandatory safety training required. Performance needs improvement.",
            ContractDecision.TERMINATE: "Contract termination recommended. Critical safety concerns.",
        }
        return descriptions.get(decision, "Unknown decision")


class DriverReportGenerator:
    """
    Generate PDF reports for driver safety evaluation.
    """
    
    def __init__(self, output_dir: str = "reports"):
        """Initialize report generator."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.scorer = DriverScorer()
    
    def generate_report(
        self, 
        df: pd.DataFrame, 
        driver_id: str = "DRIVER_001",
        driver_name: str = "Unknown Driver",
        report_period: str = None
    ) -> str:
        """
        Generate a complete PDF report for a driver.
        
        Args:
            df: DataFrame with driving features
            driver_id: Driver identifier
            driver_name: Driver's name
            report_period: Reporting period (e.g., "Q4 2024")
            
        Returns:
            Path to generated PDF
        """
        if report_period is None:
            report_period = datetime.now().strftime("%B %Y")
        
        # Compute scores
        score = self.scorer.compute_score(df)
        decision = ContractDecision.get_decision(score.overall_score)
        
        # Generate PDF
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_path = self.output_dir / f"driver_report_{driver_id}_{timestamp}.pdf"
        
        with PdfPages(pdf_path) as pdf:
            # Page 1: Executive Summary
            self._create_summary_page(pdf, driver_id, driver_name, report_period, score, decision)
            
            # Page 2: Score Breakdown
            self._create_score_breakdown_page(pdf, score)
            
            # Page 3: Driving Statistics
            self._create_statistics_page(pdf, df)
            
            # Page 4: Recommendations
            self._create_recommendations_page(pdf, score, decision)
        
        print(f"Report generated: {pdf_path}")
        return str(pdf_path)
    
    def _create_summary_page(
        self, pdf, driver_id, driver_name, report_period, score, decision
    ):
        """Create executive summary page."""
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        
        # Title
        fig.suptitle("DRIVER SAFETY EVALUATION REPORT", fontsize=16, fontweight='bold', y=0.95)
        
        # Header info
        header_text = f"""
Driver ID: {driver_id}
Driver Name: {driver_name}
Report Period: {report_period}
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}
        """
        ax.text(0.05, 0.85, header_text, fontsize=10, verticalalignment='top', 
                fontfamily='monospace', transform=ax.transAxes)
        
        # Big score display
        grade = self.scorer.score_to_grade(score.overall_score)
        score_color = ContractDecision.get_decision_color(decision)
        
        ax.text(0.5, 0.65, f"{score.overall_score:.0f}", fontsize=72, fontweight='bold',
                ha='center', va='center', color=score_color, transform=ax.transAxes)
        ax.text(0.5, 0.52, f"Grade: {grade}", fontsize=24, ha='center', va='center',
                color=score_color, transform=ax.transAxes)
        
        # Decision box
        decision_desc = ContractDecision.get_decision_description(decision)
        decision_label = decision.replace("_", " ")
        
        # Draw decision box
        from matplotlib.patches import FancyBboxPatch
        box = FancyBboxPatch((0.1, 0.32), 0.8, 0.12, 
                             boxstyle="round,pad=0.02",
                             facecolor=score_color, alpha=0.2,
                             edgecolor=score_color, linewidth=2,
                             transform=ax.transAxes)
        ax.add_patch(box)
        
        ax.text(0.5, 0.40, f"CONTRACT DECISION: {decision_label}", 
                fontsize=14, fontweight='bold', ha='center', va='center',
                color=score_color, transform=ax.transAxes)
        ax.text(0.5, 0.35, decision_desc, fontsize=10, ha='center', va='center',
                transform=ax.transAxes)
        
        # Score summary table
        summary_text = f"""
SCORE BREAKDOWN
═══════════════════════════════════════════════════════
  Behavior Score:     {score.behavior_score:>6.1f} / 100    (Weight: 40%)
  Smoothness Score:   {score.smoothness_score:>6.1f} / 100    (Weight: 25%)
  Awareness Score:    {score.awareness_score:>6.1f} / 100    (Weight: 20%)
  Speed Score:        {score.speed_score:>6.1f} / 100    (Weight: 15%)
═══════════════════════════════════════════════════════
  OVERALL SCORE:      {score.overall_score:>6.1f} / 100
═══════════════════════════════════════════════════════
        """
        ax.text(0.05, 0.25, summary_text, fontsize=9, verticalalignment='top',
                fontfamily='monospace', transform=ax.transAxes)
        
        # Footer
        ax.text(0.5, 0.02, "CONFIDENTIAL - FOR INTERNAL USE ONLY", 
                fontsize=8, ha='center', style='italic', color='gray',
                transform=ax.transAxes)
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    def _create_score_breakdown_page(self, pdf, score):
        """Create score breakdown page with charts."""
        fig, axes = plt.subplots(2, 2, figsize=(8.5, 11))
        fig.suptitle("SCORE BREAKDOWN", fontsize=14, fontweight='bold', y=0.98)
        
        # Score components bar chart
        ax1 = axes[0, 0]
        components = ['Behavior', 'Smoothness', 'Awareness', 'Speed']
        values = [score.behavior_score, score.smoothness_score, 
                  score.awareness_score, score.speed_score]
        colors = ['#FF1744' if v < 60 else '#FF9100' if v < 80 else '#00C853' for v in values]
        
        bars = ax1.barh(components, values, color=colors)
        ax1.set_xlim(0, 100)
        ax1.set_xlabel('Score')
        ax1.set_title('Component Scores')
        ax1.axvline(x=60, color='red', linestyle='--', alpha=0.5, label='Min Threshold')
        ax1.axvline(x=85, color='green', linestyle='--', alpha=0.5, label='Bonus Threshold')
        
        for bar, val in zip(bars, values):
            ax1.text(val + 2, bar.get_y() + bar.get_height()/2, 
                    f'{val:.0f}', va='center', fontsize=10)
        
        # Weights pie chart
        ax2 = axes[0, 1]
        weights = [40, 25, 20, 15]
        ax2.pie(weights, labels=components, autopct='%1.0f%%', 
                colors=['#667eea', '#764ba2', '#f093fb', '#f5576c'])
        ax2.set_title('Score Weights')
        
        # Threshold gauge
        ax3 = axes[1, 0]
        ax3.set_xlim(0, 100)
        ax3.set_ylim(0, 1)
        
        # Draw threshold zones
        ax3.axvspan(0, 50, alpha=0.3, color='red', label='Terminate')
        ax3.axvspan(50, 70, alpha=0.3, color='orange', label='Probation')
        ax3.axvspan(70, 85, alpha=0.3, color='blue', label='Retain')
        ax3.axvspan(85, 100, alpha=0.3, color='green', label='Bonus')
        
        # Mark current score
        ax3.axvline(x=score.overall_score, color='black', linewidth=3)
        ax3.scatter([score.overall_score], [0.5], s=200, color='black', zorder=5)
        ax3.text(score.overall_score, 0.7, f'{score.overall_score:.0f}', 
                ha='center', fontsize=12, fontweight='bold')
        
        ax3.set_title('Score vs Contract Thresholds')
        ax3.set_xlabel('Safety Score')
        ax3.set_yticks([])
        ax3.legend(loc='upper left', fontsize=8)
        
        # Risk factors
        ax4 = axes[1, 1]
        ax4.axis('off')
        ax4.set_title('Identified Risk Factors', fontsize=12, fontweight='bold')
        
        if score.risk_factors:
            risk_text = "\n".join([f"⚠ {rf}" for rf in score.risk_factors])
        else:
            risk_text = "✓ No significant risk factors detected"
        
        ax4.text(0.1, 0.8, risk_text, fontsize=10, verticalalignment='top',
                transform=ax4.transAxes, wrap=True)
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    def _create_statistics_page(self, pdf, df):
        """Create driving statistics page."""
        fig, axes = plt.subplots(2, 2, figsize=(8.5, 11))
        fig.suptitle("DRIVING STATISTICS", fontsize=14, fontweight='bold', y=0.98)
        
        # Speed distribution
        ax1 = axes[0, 0]
        if 'speed' in df.columns:
            ax1.hist(df['speed'].dropna(), bins=30, color='#667eea', alpha=0.7)
            ax1.axvline(df['speed'].mean(), color='red', linestyle='--', 
                       label=f'Mean: {df["speed"].mean():.1f} m/s')
            ax1.set_xlabel('Speed (m/s)')
            ax1.set_ylabel('Frequency')
            ax1.set_title('Speed Distribution')
            ax1.legend()
        
        # Steering distribution
        ax2 = axes[0, 1]
        if 'steering' in df.columns:
            ax2.hist(df['steering'].dropna(), bins=30, color='#764ba2', alpha=0.7)
            ax2.set_xlabel('Steering Angle (degrees)')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Steering Distribution')
        
        # Label distribution
        ax3 = axes[1, 0]
        if 'label' in df.columns:
            label_counts = df['label'].value_counts()
            colors = {'aggressive': '#FF1744', 'safe': '#00C853', 'drowsy': '#FF9100'}
            ax3.pie(label_counts.values, labels=label_counts.index, 
                   autopct='%1.1f%%',
                   colors=[colors.get(l, '#666') for l in label_counts.index])
            ax3.set_title('Behavior Distribution')
        
        # Summary stats table
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        stats_data = []
        for col in ['speed', 'steering', 'steering_jerk', 'radar_distance']:
            if col in df.columns:
                vals = df[col].dropna()
                stats_data.append([
                    col,
                    f"{vals.mean():.2f}",
                    f"{vals.std():.2f}",
                    f"{vals.min():.2f}",
                    f"{vals.max():.2f}"
                ])
        
        if stats_data:
            table = ax4.table(
                cellText=stats_data,
                colLabels=['Feature', 'Mean', 'Std', 'Min', 'Max'],
                loc='center',
                cellLoc='center'
            )
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1.2, 1.5)
        ax4.set_title('Feature Statistics', fontsize=12, fontweight='bold', y=0.95)
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    def _create_recommendations_page(self, pdf, score, decision):
        """Create recommendations page."""
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        
        fig.suptitle("RECOMMENDATIONS & ACTION ITEMS", fontsize=14, fontweight='bold', y=0.95)
        
        # Contract decision summary
        decision_color = ContractDecision.get_decision_color(decision)
        decision_text = f"""
CONTRACT RECOMMENDATION
{'═' * 50}

Decision: {decision.replace('_', ' ')}
{ContractDecision.get_decision_description(decision)}

        """
        ax.text(0.05, 0.88, decision_text, fontsize=11, verticalalignment='top',
                transform=ax.transAxes, color=decision_color, fontweight='bold')
        
        # Recommendations
        rec_text = "SAFETY RECOMMENDATIONS\n" + "═" * 50 + "\n\n"
        for i, rec in enumerate(score.recommendations, 1):
            rec_text += f"  {i}. {rec}\n\n"
        
        ax.text(0.05, 0.65, rec_text, fontsize=10, verticalalignment='top',
                transform=ax.transAxes)
        
        # Action items based on decision
        action_text = "REQUIRED ACTION ITEMS\n" + "═" * 50 + "\n\n"
        
        if decision == ContractDecision.TERMINATE:
            actions = [
                "Schedule immediate meeting with HR and driver",
                "Review all incident records from the evaluation period",
                "Prepare contract termination documentation",
                "Arrange vehicle handover and credential revocation",
                "Document all safety incidents for legal records"
            ]
        elif decision == ContractDecision.PROBATION:
            actions = [
                "Enroll driver in mandatory safety training program",
                "Schedule weekly performance reviews for 30 days",
                "Install additional telematics monitoring",
                "Set improvement targets with clear milestones",
                "Re-evaluate after training completion"
            ]
        elif decision == ContractDecision.RETAIN:
            actions = [
                "Continue standard monitoring procedures",
                "Share this report with driver for awareness",
                "Schedule quarterly performance review",
                "Consider for optional advanced training"
            ]
        else:  # BONUS
            actions = [
                "Process performance bonus as per company policy",
                "Recognize driver in company communications",
                "Consider for mentorship program",
                "Document as example for best practices"
            ]
        
        for i, action in enumerate(actions, 1):
            action_text += f"  ☐ {action}\n\n"
        
        ax.text(0.05, 0.40, action_text, fontsize=10, verticalalignment='top',
                transform=ax.transAxes)
        
        # Signature lines
        sig_text = """
APPROVALS
═══════════════════════════════════════════════════════

Fleet Manager: ________________________  Date: __________

HR Representative: ____________________  Date: __________

Safety Officer: _______________________  Date: __________
        """
        ax.text(0.05, 0.10, sig_text, fontsize=10, verticalalignment='top',
                transform=ax.transAxes)
        
        # Footer
        ax.text(0.5, 0.02, "CONFIDENTIAL - FOR INTERNAL USE ONLY", 
                fontsize=8, ha='center', style='italic', color='gray',
                transform=ax.transAxes)
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)


def generate_driver_report(
    data_path: str,
    output_dir: str = "reports",
    driver_id: str = "DRIVER_001",
    driver_name: str = "Test Driver"
) -> str:
    """
    Generate a PDF report for a driver.
    
    Args:
        data_path: Path to driving data CSV
        output_dir: Output directory for reports
        driver_id: Driver identifier
        driver_name: Driver's name
        
    Returns:
        Path to generated PDF
    """
    df = pd.read_csv(data_path)
    generator = DriverReportGenerator(output_dir)
    return generator.generate_report(df, driver_id, driver_name)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate driver safety report")
    parser.add_argument("--input", "-i", type=str, 
                       default="data/processed/training_data.csv",
                       help="Input CSV with driving data")
    parser.add_argument("--output", "-o", type=str, default="reports/",
                       help="Output directory for PDF reports")
    parser.add_argument("--driver-id", type=str, default="DRIVER_001",
                       help="Driver ID")
    parser.add_argument("--driver-name", type=str, default="Test Driver",
                       help="Driver name")
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("GENERATING DRIVER SAFETY REPORT")
    print("="*60)
    
    pdf_path = generate_driver_report(
        args.input,
        args.output,
        args.driver_id,
        args.driver_name
    )
    
    print(f"\n✅ Report saved to: {pdf_path}")
    print("\nContract Decision Thresholds:")
    print(f"  Score > 85: BONUS ELIGIBLE")
    print(f"  Score 70-85: RETAIN CONTRACT")
    print(f"  Score 50-70: MANDATORY TRAINING")
    print(f"  Score < 50: TERMINATE CONTRACT")
