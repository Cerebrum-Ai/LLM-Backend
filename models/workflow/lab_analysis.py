from typing import Dict, List, Optional
from datetime import datetime
import numpy as np

class LabAnalysisModel:
    """
    Lab Analysis Model for processing and interpreting laboratory results
    """
    def __init__(self):
        self.reference_ranges = {
            'WBC': {'low': 4.5, 'high': 11.0, 'unit': '10^9/L'},
            'RBC': {'low': 4.5, 'high': 5.5, 'unit': '10^12/L'},
            'HGB': {'low': 13.5, 'high': 17.5, 'unit': 'g/dL'},
            'HCT': {'low': 41.0, 'high': 50.0, 'unit': '%'},
            'PLT': {'low': 150, 'high': 450, 'unit': '10^9/L'},
            'NA': {'low': 135, 'high': 145, 'unit': 'mmol/L'},
            'K': {'low': 3.5, 'high': 5.0, 'unit': 'mmol/L'},
            'CL': {'low': 98, 'high': 107, 'unit': 'mmol/L'},
            'CO2': {'low': 23, 'high': 29, 'unit': 'mmol/L'},
            'BUN': {'low': 7, 'high': 20, 'unit': 'mg/dL'},
            'CREAT': {'low': 0.7, 'high': 1.3, 'unit': 'mg/dL'},
            'GLU': {'low': 70, 'high': 100, 'unit': 'mg/dL'},
            'CA': {'low': 8.5, 'high': 10.5, 'unit': 'mg/dL'},
            'MG': {'low': 1.7, 'high': 2.2, 'unit': 'mg/dL'},
            'PHOS': {'low': 2.5, 'high': 4.5, 'unit': 'mg/dL'},
            'AST': {'low': 10, 'high': 40, 'unit': 'U/L'},
            'ALT': {'low': 7, 'high': 56, 'unit': 'U/L'},
            'ALP': {'low': 44, 'high': 147, 'unit': 'U/L'},
            'TBIL': {'low': 0.1, 'high': 1.2, 'unit': 'mg/dL'},
            'ALB': {'low': 3.5, 'high': 5.0, 'unit': 'g/dL'},
            'TP': {'low': 6.0, 'high': 8.3, 'unit': 'g/dL'},
            'TROP': {'low': 0, 'high': 0.04, 'unit': 'ng/mL'},
            'BNP': {'low': 0, 'high': 100, 'unit': 'pg/mL'},
            'CRP': {'low': 0, 'high': 3.0, 'unit': 'mg/L'},
            'D-DIMER': {'low': 0, 'high': 0.5, 'unit': 'mg/L FEU'}
        }
        
        self.critical_values = {
            'NA': {'low': 120, 'high': 160},
            'K': {'low': 2.5, 'high': 6.5},
            'CA': {'low': 6.0, 'high': 13.0},
            'GLU': {'low': 40, 'high': 400},
            'PH': {'low': 7.2, 'high': 7.6},
            'TROP': {'high': 0.1},
            'BNP': {'high': 500}
        }
        
    def analyze_results(self, 
                       lab_results: Dict[str, float],
                       previous_results: Optional[Dict[str, List[Dict]]] = None) -> Dict:
        """
        Analyze lab results and generate interpretation
        
        Args:
            lab_results: Dictionary of current lab results
            previous_results: Optional dictionary of previous results for trend analysis
            
        Returns:
            Dict containing analysis results
        """
        # Analyze current results
        current_analysis = self._analyze_current_results(lab_results)
        
        # Analyze trends if previous results available
        trend_analysis = {}
        if previous_results:
            trend_analysis = self._analyze_trends(lab_results, previous_results)
            
        # Check for critical values
        critical_findings = self._check_critical_values(lab_results)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(current_analysis, critical_findings)
        
        return {
            'current_analysis': current_analysis,
            'trend_analysis': trend_analysis,
            'critical_findings': critical_findings,
            'recommendations': recommendations,
            'timestamp': datetime.now().isoformat()
        }
    
    def _analyze_current_results(self, results: Dict[str, float]) -> Dict[str, Dict]:
        """Analyze current lab results against reference ranges"""
        analysis = {}
        
        for test, value in results.items():
            if test in self.reference_ranges:
                ref_range = self.reference_ranges[test]
                status = 'NORMAL'
                if value < ref_range['low']:
                    status = 'LOW'
                elif value > ref_range['high']:
                    status = 'HIGH'
                    
                analysis[test] = {
                    'value': value,
                    'unit': ref_range['unit'],
                    'reference_range': f"{ref_range['low']}-{ref_range['high']}",
                    'status': status
                }
                
        return analysis
    
    def _analyze_trends(self, 
                       current_results: Dict[str, float],
                       previous_results: Dict[str, List[Dict]]) -> Dict[str, Dict]:
        """Analyze trends in lab values over time"""
        trends = {}
        
        for test, value in current_results.items():
            if test in previous_results:
                prev_values = [r['value'] for r in previous_results[test]]
                if len(prev_values) >= 2:
                    # Calculate trend
                    trend = 'STABLE'
                    if value > prev_values[-1] * 1.1:
                        trend = 'INCREASING'
                    elif value < prev_values[-1] * 0.9:
                        trend = 'DECREASING'
                        
                    trends[test] = {
                        'current_value': value,
                        'previous_value': prev_values[-1],
                        'trend': trend,
                        'percent_change': ((value - prev_values[-1]) / prev_values[-1]) * 100
                    }
                    
        return trends
    
    def _check_critical_values(self, results: Dict[str, float]) -> List[Dict]:
        """Check for critical values that require immediate attention"""
        critical_findings = []
        
        for test, value in results.items():
            if test in self.critical_values:
                crit_range = self.critical_values[test]
                if 'low' in crit_range and value < crit_range['low']:
                    critical_findings.append({
                        'test': test,
                        'value': value,
                        'type': 'CRITICALLY LOW',
                        'message': f"{test} is critically low at {value} {self.reference_ranges[test]['unit']}"
                    })
                elif 'high' in crit_range and value > crit_range['high']:
                    critical_findings.append({
                        'test': test,
                        'value': value,
                        'type': 'CRITICALLY HIGH',
                        'message': f"{test} is critically high at {value} {self.reference_ranges[test]['unit']}"
                    })
                    
        return critical_findings
    
    def _generate_recommendations(self, 
                                current_analysis: Dict[str, Dict],
                                critical_findings: List[Dict]) -> List[str]:
        """Generate recommendations based on analysis results"""
        recommendations = []
        
        # Handle critical findings first
        if critical_findings:
            recommendations.append('CRITICAL: Immediate medical attention required')
            for finding in critical_findings:
                recommendations.append(f"CRITICAL: {finding['message']}")
                
        # Handle significant abnormalities
        for test, analysis in current_analysis.items():
            if analysis['status'] in ['HIGH', 'LOW']:
                if test == 'TROP' and analysis['status'] == 'HIGH':
                    recommendations.append('Consider cardiac evaluation for elevated troponin')
                elif test == 'BNP' and analysis['status'] == 'HIGH':
                    recommendations.append('Consider heart failure evaluation for elevated BNP')
                elif test == 'D-DIMER' and analysis['status'] == 'HIGH':
                    recommendations.append('Consider evaluation for possible DVT/PE')
                elif test == 'WBC' and analysis['status'] == 'HIGH':
                    recommendations.append('Consider evaluation for possible infection')
                elif test == 'HGB' and analysis['status'] == 'LOW':
                    recommendations.append('Consider evaluation for possible anemia')
                    
        return recommendations 