from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import re
import warnings
warnings.filterwarnings('ignore')

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import transformers
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class EconomicNarrativeGenerator:
    """AI-powered economic narrative and insight generation system."""
    
    def __init__(self, openai_api_key: str = None, model_name: str = "gpt-3.5-turbo"):
        """Initialize the narrative generator."""
        self.openai_api_key = openai_api_key
        self.model_name = model_name
        
        if OPENAI_AVAILABLE and openai_api_key:
            openai.api_key = openai_api_key
            self.use_openai = True
        else:
            self.use_openai = False
            print("OpenAI not available. Using template-based narrative generation.")
        
        # Initialize sentiment analysis if available
        if TRANSFORMERS_AVAILABLE:
            try:
                self.sentiment_analyzer = pipeline(
                    "sentiment-analysis",
                    model="ProsusAI/finbert"
                )
            except Exception:
                self.sentiment_analyzer = None
        else:
            self.sentiment_analyzer = None
        
        # Economic narrative templates
        self.narrative_templates = self._load_narrative_templates()
        
        # Economic interpretation rules
        self.interpretation_rules = self._load_interpretation_rules()
    
    def _load_narrative_templates(self) -> Dict[str, str]:
        """Load narrative templates for different economic scenarios."""
        return {
            'growth_strong': """
                Economic Growth Analysis:
                The U.S. economy is showing {growth_strength} growth with GDP expanding at {gdp_rate}% annually. 
                {growth_details} This performance is {growth_comparison} historical averages.
                Key drivers include {key_drivers}. The outlook suggests {outlook}.
            """,
            
            'inflation_rising': """
                Inflation Analysis:
                Inflation pressures are {inflation_level} with the CPI rising {inflation_rate}% year-over-year. 
                {inflation_details} This trend is {inflation_trend} and impacts {inflation_impact}.
                The Federal Reserve may need to consider {policy_response}.
            """,
            
            'labor_market': """
                Labor Market Assessment:
                The unemployment rate stands at {unemployment_rate}%, indicating {labor_strength} labor market conditions.
                {employment_details} Job creation is {job_creation_trend} and labor force participation is {participation_trend}.
                Wage growth is {wage_growth}, which {wage_impact} inflation pressures.
            """,
            
            'monetary_policy': """
                Monetary Policy Analysis:
                The Federal Reserve is maintaining {policy_stance} monetary policy with the federal funds rate at {fed_rate}%.
                {policy_details} This stance is {policy_appropriateness} given current economic conditions.
                Future policy moves are likely to be {policy_outlook}.
            """,
            
            'financial_markets': """
                Financial Markets Overview:
                Equity markets are {market_performance} with the S&P 500 {market_change}% year-to-date.
                {market_details} Market volatility is {volatility_level}, indicating {volatility_interpretation}.
                Investor sentiment is {sentiment_level}, reflecting {sentiment_causes}.
            """,
            
            'recession_risk': """
                Recession Risk Assessment:
                Current recession probability is estimated at {recession_prob}%, indicating {risk_level} risk.
                {risk_factors} Leading indicators are {leading_indicator_trend}.
                The economy is {cycle_phase} with {cycle_duration} until potential turning point.
            """,
            
            'sector_analysis': """
                Sector Performance Analysis:
                {sector_performance} Key sectors showing strength include {strong_sectors}.
                {sector_details} Underperforming sectors include {weak_sectors}.
                Sector rotation suggests {rotation_implication}.
            """,
            
            'international_context': """
                Global Economic Context:
                The U.S. economy is performing {global_comparison} relative to major trading partners.
                {global_details} International trade is {trade_trend} and capital flows are {capital_flow_trend}.
                Global risks include {global_risks}.
            """
        }
    
    def _load_interpretation_rules(self) -> Dict[str, Dict]:
        """Load economic interpretation rules."""
        return {
            'gdp_growth': {
                'strong': {'min': 0.03, 'max': float('inf'), 'descriptor': 'strong'},
                'moderate': {'min': 0.01, 'max': 0.03, 'descriptor': 'moderate'},
                'weak': {'min': -0.01, 'max': 0.01, 'descriptor': 'weak'},
                'contraction': {'min': float('-inf'), 'max': -0.01, 'descriptor': 'contracting'}
            },
            
            'inflation': {
                'high': {'min': 0.04, 'max': float('inf'), 'descriptor': 'elevated'},
                'moderate': {'min': 0.025, 'max': 0.04, 'descriptor': 'moderate'},
                'low': {'min': 0.015, 'max': 0.025, 'descriptor': 'low'},
                'deflation': {'min': float('-inf'), 'max': 0.015, 'descriptor': 'deflationary'}
            },
            
            'unemployment': {
                'tight': {'min': 0, 'max': 0.04, 'descriptor': 'tight'},
                'healthy': {'min': 0.04, 'max': 0.06, 'descriptor': 'healthy'},
                'elevated': {'min': 0.06, 'max': 0.08, 'descriptor': 'elevated'},
                'high': {'min': 0.08, 'max': float('inf'), 'descriptor': 'high'}
            },
            
            'recession_risk': {
                'low': {'min': 0, 'max': 0.25, 'descriptor': 'low'},
                'moderate': {'min': 0.25, 'max': 0.50, 'descriptor': 'moderate'},
                'elevated': {'min': 0.50, 'max': 0.75, 'descriptor': 'elevated'},
                'high': {'min': 0.75, 'max': float('inf'), 'descriptor': 'high'}
            }
        }
    
    def _interpret_indicator(self, value: float, indicator_type: str) -> Tuple[str, str]:
        """Interpret economic indicator value."""
        if indicator_type not in self.interpretation_rules:
            return 'unknown', 'unknown'
        
        rules = self.interpretation_rules[indicator_type]
        
        for level, rule in rules.items():
            if rule['min'] <= value < rule['max']:
                return level, rule['descriptor']
        
        return 'unknown', 'unknown'
    
    def _generate_insight_summary(self, economic_data: Dict[str, Any]) -> List[str]:
        """Generate key insights from economic data."""
        insights = []
        
        # GDP insights
        if 'gdp_growth' in economic_data:
            gdp_level, gdp_desc = self._interpret_indicator(economic_data['gdp_growth'], 'gdp_growth')
            if gdp_level == 'strong':
                insights.append("Robust economic growth exceeding historical averages")
            elif gdp_level == 'contraction':
                insights.append("Economy contracting, signaling potential recession")
        
        # Inflation insights
        if 'inflation_rate' in economic_data:
            inf_level, inf_desc = self._interpret_indicator(economic_data['inflation_rate'], 'inflation')
            if inf_level == 'high':
                insights.append("Elevated inflation pressures requiring policy attention")
            elif inf_level == 'deflation':
                insights.append("Deflationary risks present, requiring accommodative policy")
        
        # Labor market insights
        if 'unemployment_rate' in economic_data:
            emp_level, emp_desc = self._interpret_indicator(economic_data['unemployment_rate'], 'unemployment')
            if emp_level == 'tight':
                insights.append("Tight labor market supporting wage growth")
            elif emp_level == 'high':
                insights.append("Weak labor market indicating economic distress")
        
        # Recession risk insights
        if 'recession_probability' in economic_data:
            risk_level, risk_desc = self._interpret_indicator(economic_data['recession_probability'], 'recession_risk')
            if risk_level in ['elevated', 'high']:
                insights.append(f"High recession probability ({risk_desc}) requires risk management")
        
        return insights
    
    def generate_narrative_section(
        self,
        section_type: str,
        economic_data: Dict[str, Any],
        use_ai: bool = None
    ) -> str:
        """Generate narrative for specific economic section."""
        
        use_ai = use_ai if use_ai is not None else self.use_openai
        
        if use_ai and OPENAI_AVAILABLE:
            return self._generate_ai_narrative(section_type, economic_data)
        else:
            return self._generate_template_narrative(section_type, economic_data)
    
    def _generate_template_narrative(self, section_type: str, economic_data: Dict[str, Any]) -> str:
        """Generate narrative using templates."""
        
        if section_type not in self.narrative_templates:
            return f"No narrative template available for {section_type}"
        
        template = self.narrative_templates[section_type]
        
        # Fill template with data
        try:
            if section_type == 'growth_strong':
                gdp_growth = economic_data.get('gdp_growth', 0)
                gdp_level, gdp_desc = self._interpret_indicator(gdp_growth, 'gdp_growth')
                
                narrative = template.format(
                    growth_strength=gdp_desc,
                    gdp_rate=f"{gdp_growth:.1%}",
                    growth_details=self._get_growth_details(economic_data),
                    growth_comparison=self._get_growth_comparison(gdp_growth),
                    key_drivers=economic_data.get('growth_drivers', 'consumer spending and business investment'),
                    outlook=economic_data.get('growth_outlook', 'continued expansion')
                )
            
            elif section_type == 'inflation_rising':
                inflation_rate = economic_data.get('inflation_rate', 0)
                inf_level, inf_desc = self._interpret_indicator(inflation_rate, 'inflation')
                
                narrative = template.format(
                    inflation_level=inf_desc,
                    inflation_rate=f"{inflation_rate:.1%}",
                    inflation_details=self._get_inflation_details(economic_data),
                    inflation_trend=self._get_inflation_trend(economic_data),
                    inflation_impact='household purchasing power and business costs',
                    policy_response=self._get_policy_response(inf_level)
                )
            
            elif section_type == 'labor_market':
                unemployment_rate = economic_data.get('unemployment_rate', 0.05)
                emp_level, emp_desc = self._interpret_indicator(unemployment_rate, 'unemployment')
                
                narrative = template.format(
                    unemployment_rate=f"{unemployment_rate:.1%}",
                    labor_strength=emp_desc,
                    employment_details=self._get_employment_details(economic_data),
                    job_creation_trend=economic_data.get('job_creation_trend', 'steady'),
                    participation_trend=economic_data.get('participation_trend', 'stable'),
                    wage_growth=economic_data.get('wage_growth', 'moderate'),
                    wage_impact='contributing to' if inflation_rate > 0.03 else 'not significantly impacting'
                )
            
            elif section_type == 'recession_risk':
                recession_prob = economic_data.get('recession_probability', 0.25)
                risk_level, risk_desc = self._interpret_indicator(recession_prob, 'recession_risk')
                
                narrative = template.format(
                    recession_prob=f"{recession_prob:.0%}",
                    risk_level=risk_desc,
                    risk_factors=self._get_risk_factors(economic_data),
                    leading_indicator_trend=economic_data.get('leading_indicators', 'mixed'),
                    cycle_phase=economic_data.get('cycle_phase', 'late expansion'),
                    cycle_duration='12-18 months'
                )
            
            else:
                narrative = template.format(**economic_data)
            
            return narrative.strip()
            
        except Exception as e:
            return f"Error generating narrative for {section_type}: {str(e)}"
    
    def _generate_ai_narrative(self, section_type: str, economic_data: Dict[str, Any]) -> str:
        """Generate narrative using AI (OpenAI)."""
        
        if not OPENAI_AVAILABLE:
            return self._generate_template_narrative(section_type, economic_data)
        
        try:
            prompt = self._create_ai_prompt(section_type, economic_data)
            
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are an expert economic analyst providing clear, insightful economic commentary."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"AI narrative generation failed: {e}")
            return self._generate_template_narrative(section_type, economic_data)
    
    def _create_ai_prompt(self, section_type: str, economic_data: Dict[str, Any]) -> str:
        """Create AI prompt for narrative generation."""
        
        data_summary = "\n".join([f"- {k}: {v}" for k, v in economic_data.items()])
        
        prompts = {
            'growth_strong': f"""
                Analyze the following economic data and provide a comprehensive growth analysis:
                {data_summary}
                
                Focus on:
                - Current growth trajectory and sustainability
                - Key growth drivers and headwinds
                - Comparison to historical trends
                - Forward-looking outlook
            """,
            
            'inflation_rising': f"""
                Analyze the inflation situation based on this data:
                {data_summary}
                
                Focus on:
                - Current inflation level and trend
                - Underlying inflation drivers
                - Impact on households and businesses
                - Appropriate policy responses
            """,
            
            'labor_market': f"""
                Assess labor market conditions with this data:
                {data_summary}
                
                Focus on:
                - Current employment situation
                - Wage growth trends
                - Labor force participation
                - Implications for monetary policy
            """,
            
            'recession_risk': f"""
                Evaluate recession risk based on these indicators:
                {data_summary}
                
                Focus on:
                - Current recession probability
                - Key risk factors
                - Leading indicator trends
                - Economic cycle position
            """
        }
        
        return prompts.get(section_type, f"Analyze this economic data: {data_summary}")
    
    def _get_growth_details(self, data: Dict[str, Any]) -> str:
        """Get detailed growth analysis."""
        gdp_growth = data.get('gdp_growth', 0)
        
        if gdp_growth > 0.03:
            return "Growth is being driven by strong consumer spending and business investment."
        elif gdp_growth > 0:
            return "Growth remains positive but shows signs of moderation."
        else:
            return "The economy is contracting, with weakness across multiple sectors."
    
    def _get_growth_comparison(self, gdp_growth: float) -> str:
        """Compare growth to historical averages."""
        if gdp_growth > 0.025:
            return "above"
        elif gdp_growth > 0.015:
            return "near"
        else:
            return "below"
    
    def _get_inflation_details(self, data: Dict[str, Any]) -> str:
        """Get inflation details."""
        inflation_rate = data.get('inflation_rate', 0)
        
        if inflation_rate > 0.04:
            return "Inflation is running well above the Federal Reserve's 2% target."
        elif inflation_rate > 0.025:
            return "Inflation remains elevated but is showing signs of moderation."
        else:
            return "Inflation is contained within the Fed's target range."
    
    def _get_inflation_trend(self, data: Dict[str, Any]) -> str:
        """Get inflation trend."""
        inflation_trend = data.get('inflation_trend', 'stable')
        return f"{inflation_trend} and requires close monitoring"
    
    def _get_policy_response(self, inflation_level: str) -> str:
        """Get appropriate policy response."""
        responses = {
            'high': "continued rate hikes to bring inflation back to target",
            'moderate': "a balanced approach with rates at current levels",
            'low': "potential rate cuts to support economic activity",
            'deflation': "aggressive monetary stimulus to avoid deflationary spiral"
        }
        return responses.get(inflation_level, "careful calibration of policy stance")
    
    def _get_employment_details(self, data: Dict[str, Any]) -> str:
        """Get employment details."""
        unemployment_rate = data.get('unemployment_rate', 0.05)
        
        if unemployment_rate < 0.04:
            return "The labor market remains tight with more job openings than unemployed workers."
        elif unemployment_rate < 0.06:
            return "Employment conditions are balanced with steady job creation."
        else:
            return "Labor market weakness is evident with declining job growth."
    
    def _get_risk_factors(self, data: Dict[str, Any]) -> str:
        """Get key risk factors."""
        risks = []
        
        if data.get('inflation_rate', 0) > 0.04:
            risks.append("elevated inflation")
        if data.get('unemployment_rate', 0.05) > 0.06:
            risks.append("rising unemployment")
        if data.get('yield_curve_inversion', False):
            risks.append("yield curve inversion")
        
        if risks:
            return ", ".join(risks)
        else:
            return "moderate risk levels with no immediate threats"
    
    def generate_comprehensive_report(
        self,
        economic_data: Dict[str, Any],
        forecasts: Dict[str, pd.DataFrame] = None,
        include_sections: List[str] = None
    ) -> Dict[str, str]:
        """Generate comprehensive economic narrative report."""
        
        if include_sections is None:
            include_sections = [
                'growth_strong', 'inflation_rising', 'labor_market',
                'monetary_policy', 'recession_risk'
            ]
        
        report = {
            'executive_summary': self._generate_executive_summary(economic_data),
            'key_insights': self._generate_insight_summary(economic_data),
            'sections': {},
            'forecast_analysis': self._generate_forecast_analysis(forecasts) if forecasts else "",
            'recommendations': self._generate_recommendations(economic_data),
            'risk_assessment': self._generate_risk_assessment(economic_data)
        }
        
        # Generate individual sections
        for section in include_sections:
            report['sections'][section] = self.generate_narrative_section(section, economic_data)
        
        return report
    
    def _generate_executive_summary(self, economic_data: Dict[str, Any]) -> str:
        """Generate executive summary."""
        
        gdp_growth = economic_data.get('gdp_growth', 0)
        inflation_rate = economic_data.get('inflation_rate', 0)
        unemployment_rate = economic_data.get('unemployment_rate', 0.05)
        recession_prob = economic_data.get('recession_probability', 0.25)
        
        gdp_level, gdp_desc = self._interpret_indicator(gdp_growth, 'gdp_growth')
        inf_level, inf_desc = self._interpret_indicator(inflation_rate, 'inflation')
        
        summary = f"""
        Executive Summary:
        
        The U.S. economy is experiencing {gdp_desc} growth with GDP expanding at {gdp_growth:.1%} annually. 
        Inflation remains {inf_desc} at {inflation_rate:.1%}, while the unemployment rate stands at {unemployment_rate:.1%}. 
        The current recession probability is estimated at {recession_prob:.0%}, indicating {self._interpret_indicator(recession_prob, 'recession_risk')[1]} risk levels.
        
        Key economic indicators suggest {self._get_overall_assessment(economic_data)} economic conditions.
        Policy makers should focus on {self._get_policy_priorities(economic_data)}.
        """
        
        return summary.strip()
    
    def _generate_forecast_analysis(self, forecasts: Dict[str, pd.DataFrame]) -> str:
        """Generate analysis of forecasts."""
        
        analysis = "Forecast Analysis:\n\n"
        
        for indicator, forecast_df in forecasts.items():
            if not forecast_df.empty:
                latest_forecast = forecast_df['yhat'].iloc[0]
                forecast_trend = "increasing" if len(forecast_df) > 1 and forecast_df['yhat'].iloc[-1] > latest_forecast else "decreasing"
                
                analysis += f"{indicator}: Forecast to be {forecast_trend} over the forecast horizon. "
                analysis += f"Expected value in next period: {latest_forecast:.2f}\n"
        
        return analysis
    
    def _generate_recommendations(self, economic_data: Dict[str, Any]) -> str:
        """Generate policy recommendations."""
        
        recommendations = []
        
        gdp_growth = economic_data.get('gdp_growth', 0)
        inflation_rate = economic_data.get('inflation_rate', 0)
        unemployment_rate = economic_data.get('unemployment_rate', 0.05)
        
        if inflation_rate > 0.04:
            recommendations.append("Maintain restrictive monetary policy until inflation shows sustained decline")
        elif gdp_growth < 0.01:
            recommendations.append("Consider accommodative policy measures to support economic growth")
        
        if unemployment_rate > 0.07:
            recommendations.append("Implement targeted fiscal measures to support labor market recovery")
        
        if not recommendations:
            recommendations.append("Current policy stance appears appropriate for prevailing economic conditions")
        
        return "Policy Recommendations:\n" + "\n".join(f"- {rec}" for rec in recommendations)
    
    def _generate_risk_assessment(self, economic_data: Dict[str, Any]) -> str:
        """Generate risk assessment."""
        
        risks = []
        mitigations = []
        
        if economic_data.get('inflation_rate', 0) > 0.04:
            risks.append("Persistent inflation could erode purchasing power")
            mitigations.append("Continued monetary policy tightening")
        
        if economic_data.get('recession_probability', 0) > 0.5:
            risks.append("Elevated recession risk could trigger financial market volatility")
            mitigations.append("Build fiscal buffers and strengthen social safety nets")
        
        if economic_data.get('yield_curve_inversion', False):
            risks.append("Yield curve inversion historically precedes recessions")
            mitigations.append("Monitor leading indicators closely")
        
        risk_assessment = "Risk Assessment:\n\n"
        
        if risks:
            risk_assessment += "Key Risks:\n" + "\n".join(f"- {risk}" for risk in risks)
            risk_assessment += "\n\nMitigation Strategies:\n" + "\n".join(f"- {mitigation}" for mitigation in mitigations)
        else:
            risk_assessment += "Economic risks appear contained at current levels."
        
        return risk_assessment
    
    def _get_overall_assessment(self, economic_data: Dict[str, Any]) -> str:
        """Get overall economic assessment."""
        
        gdp_growth = economic_data.get('gdp_growth', 0)
        inflation_rate = economic_data.get('inflation_rate', 0)
        unemployment_rate = economic_data.get('unemployment_rate', 0.05)
        
        if gdp_growth > 0.025 and inflation_rate < 0.03 and unemployment_rate < 0.05:
            return "favorable"
        elif gdp_growth > 0 and inflation_rate < 0.04 and unemployment_rate < 0.07:
            return "moderate"
        else:
            return "challenging"
    
    def _get_policy_priorities(self, economic_data: Dict[str, Any]) -> str:
        """Get policy priorities."""
        
        inflation_rate = economic_data.get('inflation_rate', 0)
        gdp_growth = economic_data.get('gdp_growth', 0)
        
        if inflation_rate > 0.04:
            return "inflation control and price stability"
        elif gdp_growth < 0.01:
            return "economic stimulus and growth support"
        else:
            return "maintaining balanced policy approach"
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of economic text."""
        
        if self.sentiment_analyzer:
            try:
                result = self.sentiment_analyzer(text)[0]
                return {
                    'sentiment': result['label'],
                    'confidence': result['score']
                }
            except Exception:
                pass
        
        # Fallback simple sentiment analysis
        positive_words = ['growth', 'strong', 'recovery', 'expansion', 'increase']
        negative_words = ['recession', 'decline', 'contraction', 'weakness', 'decrease']
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            sentiment = 'positive'
        elif negative_count > positive_count:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        return {
            'sentiment': sentiment,
            'confidence': min(abs(positive_count - negative_count) / max(positive_count + negative_count, 1), 1.0)
        }
