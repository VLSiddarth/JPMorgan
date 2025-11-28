"""
PDF Report Generator
Creates professional PDF reports for portfolio analysis and market research
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer,
    PageBreak, Image, KeepTogether
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from io import BytesIO

logger = logging.getLogger(__name__)
sns.set_style("whitegrid")


class PDFReportGenerator:
    """Generate professional PDF reports"""
    
    def __init__(self, company_name: str = "JPMorgan Asset Management"):
        """
        Initialize PDF generator
        
        Args:
            company_name: Company name for headers
        """
        self.company_name = company_name
        self.styles = getSampleStyleSheet()
        self._create_custom_styles()
        
    def _create_custom_styles(self):
        """Create custom paragraph styles"""
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1f77b4'),
            spaceAfter=30,
            alignment=TA_CENTER
        ))
        
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#2c3e50'),
            spaceBefore=20,
            spaceAfter=12,
            borderWidth=2,
            borderColor=colors.HexColor('#1f77b4'),
            borderPadding=5
        ))
        
        self.styles.add(ParagraphStyle(
            name='BodyText',
            parent=self.styles['BodyText'],
            fontSize=11,
            leading=14,
            spaceAfter=12
        ))
        
        self.styles.add(ParagraphStyle(
            name='Footer',
            parent=self.styles['Normal'],
            fontSize=8,
            textColor=colors.grey,
            alignment=TA_CENTER
        ))
    
    def generate_portfolio_report(self,
                                 portfolio_data: Dict[str, Any],
                                 output_path: str,
                                 include_charts: bool = True) -> bool:
        """
        Generate comprehensive portfolio report
        
        Args:
            portfolio_data: Portfolio data dictionary
            output_path: Output file path
            include_charts: Whether to include charts
            
        Returns:
            True if successful
        """
        try:
            doc = SimpleDocTemplate(output_path, pagesize=letter,
                                  topMargin=0.5*inch, bottomMargin=0.5*inch)
            story = []
            
            # Title page
            story.extend(self._create_title_page(
                "Portfolio Analysis Report",
                portfolio_data.get('portfolio_name', 'Portfolio')
            ))
            story.append(PageBreak())
            
            # Executive Summary
            story.append(Paragraph("Executive Summary", self.styles['SectionHeader']))
            summary_text = self._create_executive_summary(portfolio_data)
            story.append(Paragraph(summary_text, self.styles['BodyText']))
            story.append(Spacer(1, 0.3*inch))
            
            # Performance Metrics
            story.append(Paragraph("Performance Metrics", self.styles['SectionHeader']))
            if 'metrics' in portfolio_data:
                metrics_table = self._create_metrics_table(portfolio_data['metrics'])
                story.append(metrics_table)
            story.append(Spacer(1, 0.2*inch))
            
            # Holdings
            if 'holdings' in portfolio_data:
                story.append(Paragraph("Current Holdings", self.styles['SectionHeader']))
                holdings_table = self._create_holdings_table(portfolio_data['holdings'])
                story.append(holdings_table)
                story.append(Spacer(1, 0.2*inch))
            
            # Charts
            if include_charts and 'returns' in portfolio_data:
                story.append(PageBreak())
                story.append(Paragraph("Performance Analysis", self.styles['SectionHeader']))
                
                # Returns chart
                returns_img = self._create_returns_chart(portfolio_data['returns'])
                if returns_img:
                    story.append(Image(returns_img, width=6*inch, height=4*inch))
                story.append(Spacer(1, 0.2*inch))
                
                # Allocation chart
                if 'allocation' in portfolio_data:
                    alloc_img = self._create_allocation_chart(portfolio_data['allocation'])
                    if alloc_img:
                        story.append(Image(alloc_img, width=5*inch, height=4*inch))
            
            # Risk Analysis
            if 'risk_metrics' in portfolio_data:
                story.append(PageBreak())
                story.append(Paragraph("Risk Analysis", self.styles['SectionHeader']))
                risk_table = self._create_risk_table(portfolio_data['risk_metrics'])
                story.append(risk_table)
            
            # Footer
            story.append(PageBreak())
            story.extend(self._create_footer())
            
            # Build PDF
            doc.build(story)
            logger.info(f"Portfolio report generated: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to generate portfolio report: {e}")
            return False
    
    def generate_research_report(self,
                                research_data: Dict[str, Any],
                                output_path: str) -> bool:
        """
        Generate market research report (e.g., European Stocks Comeback)
        
        Args:
            research_data: Research data and findings
            output_path: Output file path
            
        Returns:
            True if successful
        """
        try:
            doc = SimpleDocTemplate(output_path, pagesize=letter)
            story = []
            
            # Title
            title = research_data.get('title', 'Market Research Report')
            story.extend(self._create_title_page(title, research_data.get('subtitle', '')))
            story.append(PageBreak())
            
            # Executive Summary
            story.append(Paragraph("Executive Summary", self.styles['SectionHeader']))
            summary = research_data.get('executive_summary', '')
            for para in summary.split('\n\n'):
                if para.strip():
                    story.append(Paragraph(para, self.styles['BodyText']))
            story.append(Spacer(1, 0.3*inch))
            
            # Key Findings
            if 'key_findings' in research_data:
                story.append(Paragraph("Key Findings", self.styles['SectionHeader']))
                for i, finding in enumerate(research_data['key_findings'], 1):
                    text = f"<b>{i}.</b> {finding}"
                    story.append(Paragraph(text, self.styles['BodyText']))
                story.append(Spacer(1, 0.3*inch))
            
            # Detailed Analysis Sections
            if 'sections' in research_data:
                for section in research_data['sections']:
                    story.append(PageBreak())
                    story.append(Paragraph(section['title'], self.styles['SectionHeader']))
                    story.append(Paragraph(section['content'], self.styles['BodyText']))
                    
                    # Add table if present
                    if 'table' in section:
                        table = self._create_data_table(section['table'])
                        story.append(table)
                    
                    story.append(Spacer(1, 0.2*inch))
            
            # Investment Recommendations
            if 'recommendations' in research_data:
                story.append(PageBreak())
                story.append(Paragraph("Investment Recommendations", 
                                      self.styles['SectionHeader']))
                for rec in research_data['recommendations']:
                    story.append(Paragraph(f"<b>{rec['title']}</b>", 
                                          self.styles['BodyText']))
                    story.append(Paragraph(rec['description'], 
                                          self.styles['BodyText']))
                    story.append(Spacer(1, 0.1*inch))
            
            # Risk Factors
            if 'risks' in research_data:
                story.append(PageBreak())
                story.append(Paragraph("Risk Factors", self.styles['SectionHeader']))
                for risk in research_data['risks']:
                    story.append(Paragraph(f"<b>{risk['title']}</b> ({risk['severity']})", 
                                          self.styles['BodyText']))
                    story.append(Paragraph(risk['description'], 
                                          self.styles['BodyText']))
                    story.append(Spacer(1, 0.1*inch))
            
            story.extend(self._create_footer())
            doc.build(story)
            logger.info(f"Research report generated: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to generate research report: {e}")
            return False
    
    def _create_title_page(self, title: str, subtitle: str) -> List:
        """Create title page elements"""
        elements = []
        
        elements.append(Spacer(1, 2*inch))
        elements.append(Paragraph(self.company_name, self.styles['Title']))
        elements.append(Spacer(1, 0.5*inch))
        elements.append(Paragraph(title, self.styles['CustomTitle']))
        
        if subtitle:
            elements.append(Spacer(1, 0.2*inch))
            elements.append(Paragraph(subtitle, self.styles['Heading3']))
        
        elements.append(Spacer(1, 0.5*inch))
        date_str = datetime.now().strftime("%B %d, %Y")
        elements.append(Paragraph(date_str, self.styles['Normal']))
        
        return elements
    
    def _create_executive_summary(self, data: Dict) -> str:
        """Create executive summary text"""
        metrics = data.get('metrics', {})
        
        summary = f"""
        This report provides a comprehensive analysis of the portfolio performance 
        for the period ending {datetime.now().strftime('%B %Y')}. 
        """
        
        if 'total_return' in metrics:
            summary += f"The portfolio achieved a total return of {metrics['total_return']:.2%}. "
        
        if 'sharpe_ratio' in metrics:
            summary += f"The Sharpe ratio stands at {metrics['sharpe_ratio']:.2f}, "
            summary += "indicating risk-adjusted performance. "
        
        return summary
    
    def _create_metrics_table(self, metrics: Dict) -> Table:
        """Create performance metrics table"""
        data = [['Metric', 'Value']]
        
        metric_labels = {
            'total_return': 'Total Return',
            'ytd_return': 'YTD Return',
            'volatility': 'Volatility (Ann.)',
            'sharpe_ratio': 'Sharpe Ratio',
            'max_drawdown': 'Maximum Drawdown',
            'var_95': '95% VaR',
            'beta': 'Beta'
        }
        
        for key, label in metric_labels.items():
            if key in metrics:
                value = metrics[key]
                if key in ['total_return', 'ytd_return', 'volatility', 'max_drawdown', 'var_95']:
                    formatted = f"{value:.2%}"
                else:
                    formatted = f"{value:.2f}"
                data.append([label, formatted])
        
        table = Table(data, colWidths=[3*inch, 2*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f77b4')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f0f0')])
        ]))
        
        return table
    
    def _create_holdings_table(self, holdings: pd.DataFrame) -> Table:
        """Create holdings table"""
        # Convert DataFrame to list format
        data = [holdings.columns.tolist()]
        for _, row in holdings.iterrows():
            data.append(row.tolist())
        
        table = Table(data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f77b4')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f0f0')])
        ]))
        
        return table
    
    def _create_returns_chart(self, returns: pd.Series) -> Optional[BytesIO]:
        """Create returns chart"""
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            cumulative = (1 + returns).cumprod()
            ax.plot(cumulative.index, cumulative.values, linewidth=2)
            ax.set_title('Cumulative Returns', fontsize=14, fontweight='bold')
            ax.set_xlabel('Date')
            ax.set_ylabel('Cumulative Return')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            
            img_buffer = BytesIO()
            plt.savefig(img_buffer, format='png', dpi=150)
            plt.close()
            img_buffer.seek(0)
            return img_buffer
        except Exception as e:
            logger.error(f"Failed to create returns chart: {e}")
            return None
    
    def _create_allocation_chart(self, allocation: Dict) -> Optional[BytesIO]:
        """Create allocation pie chart"""
        try:
            fig, ax = plt.subplots(figsize=(8, 8))
            labels = list(allocation.keys())
            sizes = list(allocation.values())
            
            ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
            ax.set_title('Portfolio Allocation', fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            img_buffer = BytesIO()
            plt.savefig(img_buffer, format='png', dpi=150)
            plt.close()
            img_buffer.seek(0)
            return img_buffer
        except Exception as e:
            logger.error(f"Failed to create allocation chart: {e}")
            return None
    
    def _create_risk_table(self, risk_metrics: Dict) -> Table:
        """Create risk metrics table"""
        data = [['Risk Metric', 'Value']]
        
        for key, value in risk_metrics.items():
            formatted_key = key.replace('_', ' ').title()
            if isinstance(value, float):
                formatted_value = f"{value:.2%}" if abs(value) < 1 else f"{value:.2f}"
            else:
                formatted_value = str(value)
            data.append([formatted_key, formatted_value])
        
        table = Table(data, colWidths=[3*inch, 2*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#dc3545')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        return table
    
    def _create_data_table(self, table_data: Dict) -> Table:
        """Create generic data table"""
        data = [table_data.get('headers', [])]
        data.extend(table_data.get('rows', []))
        
        table = Table(data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        return table
    
    def _create_footer(self) -> List:
        """Create footer elements"""
        elements = []
        elements.append(Spacer(1, 0.5*inch))
        
        disclaimer = """
        <i>This report is for informational purposes only and does not constitute 
        investment advice. Past performance does not guarantee future results.</i>
        """
        elements.append(Paragraph(disclaimer, self.styles['Footer']))
        
        footer_text = f"""
        Generated by {self.company_name} | {datetime.now().strftime('%Y-%m-%d %H:%M')}
        """
        elements.append(Paragraph(footer_text, self.styles['Footer']))
        
        return elements