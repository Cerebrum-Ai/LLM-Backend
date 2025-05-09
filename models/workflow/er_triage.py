from typing import Dict, List, Optional
import numpy as np
from datetime import datetime

class ERTriageModel:
    """
    ER Triage Model for patient assessment and prioritization
    """
    def __init__(self):
        self.triage_levels = {
            'RESUSCITATION': 1,  # Immediate, life-saving intervention
            'EMERGENT': 2,       # Very urgent, potential life-threatening
            'URGENT': 3,         # Urgent, but not immediately life-threatening
            'SEMI-URGENT': 4,    # Needs care within 1-2 hours
            'NON-URGENT': 5      # Can wait for care
        }
        
    def assess_patient(self, 
                      vitals: Dict[str, float],
                      symptoms: List[str],
                      medical_history: Optional[Dict] = None) -> Dict:
        """
        Assess patient condition and assign triage level
        
        Args:
            vitals: Dictionary of vital signs
            symptoms: List of reported symptoms
            medical_history: Optional medical history
            
        Returns:
            Dict containing triage assessment
        """
        # Calculate severity score
        severity_score = self._calculate_severity(vitals, symptoms)
        
        # Determine triage level
        triage_level = self._determine_triage_level(severity_score)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(triage_level, symptoms)
        
        return {
            'triage_level': triage_level,
            'severity_score': severity_score,
            'recommendations': recommendations,
            'timestamp': datetime.now().isoformat(),
            'critical_findings': self._check_critical_findings(vitals, symptoms)
        }
    
    def _calculate_severity(self, vitals: Dict[str, float], symptoms: List[str]) -> float:
        """Calculate severity score based on vitals and symptoms"""
        score = 0.0
        
        # Check vitals
        if vitals.get('heart_rate', 0) > 120 or vitals.get('heart_rate', 0) < 50:
            score += 2.0
        if vitals.get('blood_pressure_systolic', 0) > 180 or vitals.get('blood_pressure_systolic', 0) < 90:
            score += 2.0
        if vitals.get('oxygen_saturation', 0) < 92:
            score += 2.0
        if vitals.get('temperature', 0) > 38.5 or vitals.get('temperature', 0) < 35:
            score += 1.5
            
        # Check symptoms
        critical_symptoms = ['chest pain', 'shortness of breath', 'severe bleeding', 
                           'altered mental status', 'seizure']
        for symptom in symptoms:
            if symptom.lower() in critical_symptoms:
                score += 2.0
                
        return min(10.0, score)  # Cap at 10
    
    def _determine_triage_level(self, severity_score: float) -> str:
        """Determine triage level based on severity score"""
        if severity_score >= 8:
            return 'RESUSCITATION'
        elif severity_score >= 6:
            return 'EMERGENT'
        elif severity_score >= 4:
            return 'URGENT'
        elif severity_score >= 2:
            return 'SEMI-URGENT'
        else:
            return 'NON-URGENT'
    
    def _generate_recommendations(self, triage_level: str, symptoms: List[str]) -> List[str]:
        """Generate recommendations based on triage level and symptoms"""
        recommendations = []
        
        if triage_level in ['RESUSCITATION', 'EMERGENT']:
            recommendations.append('Immediate medical attention required')
            recommendations.append('Prepare resuscitation equipment')
            if 'chest pain' in symptoms:
                recommendations.append('Prepare ECG and cardiac monitoring')
            if 'shortness of breath' in symptoms:
                recommendations.append('Prepare oxygen and respiratory support')
                
        elif triage_level == 'URGENT':
            recommendations.append('Urgent medical attention required')
            recommendations.append('Monitor vital signs every 15 minutes')
            
        elif triage_level == 'SEMI-URGENT':
            recommendations.append('Medical attention required within 1-2 hours')
            recommendations.append('Monitor vital signs every 30 minutes')
            
        else:
            recommendations.append('Routine medical attention')
            recommendations.append('Monitor vital signs every hour')
            
        return recommendations
    
    def _check_critical_findings(self, vitals: Dict[str, float], symptoms: List[str]) -> List[str]:
        """Check for critical findings that require immediate attention"""
        critical_findings = []
        
        # Check vitals
        if vitals.get('heart_rate', 0) > 150:
            critical_findings.append('Severe tachycardia')
        if vitals.get('heart_rate', 0) < 40:
            critical_findings.append('Severe bradycardia')
        if vitals.get('blood_pressure_systolic', 0) > 200:
            critical_findings.append('Severe hypertension')
        if vitals.get('blood_pressure_systolic', 0) < 80:
            critical_findings.append('Severe hypotension')
        if vitals.get('oxygen_saturation', 0) < 90:
            critical_findings.append('Severe hypoxemia')
            
        # Check symptoms
        if 'chest pain' in symptoms and vitals.get('heart_rate', 0) > 100:
            critical_findings.append('Possible acute coronary syndrome')
        if 'shortness of breath' in symptoms and vitals.get('oxygen_saturation', 0) < 92:
            critical_findings.append('Possible respiratory failure')
            
        return critical_findings 