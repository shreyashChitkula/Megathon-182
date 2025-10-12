"""
Text Attention Visualization Module
Highlights important words in user's claim description with color-coded annotations
"""

import re
from typing import Dict, List, Tuple


class TextAttentionAnalyzer:
    """
    Analyzes and visualizes important keywords in claim descriptions
    Provides color-coded highlighting for better interpretability
    """
    
    def __init__(self):
        # Part keywords mapping
        self.part_keywords = {
            'bonnet': ['bonnet', 'hood', 'engine cover'],
            'bumper': ['bumper', 'fender bar', 'front bar', 'rear bar'],
            'dickey': ['dickey', 'trunk', 'boot', 'cargo'],
            'door': ['door', 'doors', 'front door', 'rear door', 'driver door'],
            'fender': ['fender', 'wheel arch', 'quarter panel'],
            'light': ['light', 'headlight', 'taillight', 'lamp', 'headlamp'],
            'windshield': ['windshield', 'windscreen', 'glass', 'window']
        }
        
        # Severity keywords
        self.severity_keywords = {
            'severe': ['severe', 'major', 'extensive', 'serious', 'critical', 'badly', 'heavily'],
            'moderate': ['moderate', 'medium', 'considerable', 'significant', 'damaged'],
            'minor': ['minor', 'small', 'slight', 'light', 'scratched', 'dent', 'scratch']
        }
        
        # Action/damage keywords
        self.damage_keywords = ['damaged', 'broken', 'cracked', 'dented', 'scratched', 
                               'shattered', 'bent', 'torn', 'crushed', 'smashed']
    
    def analyze_text_attention(self, description: str, detected_parts: List[str]) -> Dict:
        """
        Analyze text and generate attention highlighting
        
        Args:
            description: User's claim description
            detected_parts: List of parts detected by YOLO
            
        Returns:
            dict: Attention analysis with highlighted HTML, statistics, and insights
        """
        if not description:
            return {
                'highlighted_html': '<p style="color: #718096;">No description provided</p>',
                'statistics': {},
                'insights': []
            }
        
        # Normalize
        description_lower = description.lower()
        detected_parts_lower = [p.lower() for p in detected_parts]
        
        # Find all keyword matches
        matches = self._find_matches(description_lower, detected_parts_lower)
        
        # Generate highlighted HTML
        highlighted_html = self._generate_highlighted_html(description, matches)
        
        # Calculate statistics
        statistics = self._calculate_statistics(matches, detected_parts_lower)
        
        # Generate insights
        insights = self._generate_insights(matches, detected_parts_lower, description_lower)
        
        return {
            'highlighted_html': highlighted_html,
            'statistics': statistics,
            'insights': insights,
            'matches': matches
        }
    
    def _find_matches(self, text: str, detected_parts: List[str]) -> Dict[str, List[Tuple[int, int, str]]]:
        """
        Find all keyword matches in text
        
        Returns:
            dict: Category -> List of (start_pos, end_pos, matched_word, match_type)
        """
        matches = {
            'matched_parts': [],      # Parts mentioned AND detected
            'unmatched_parts': [],    # Parts mentioned but NOT detected
            'severity': [],           # Severity keywords
            'damage': []              # Damage action words
        }
        
        words = re.findall(r'\b\w+\b', text)
        current_pos = 0
        
        for word in words:
            word_lower = word.lower()
            start_pos = text.find(word, current_pos)
            end_pos = start_pos + len(word)
            current_pos = end_pos
            
            # Check if it's a part keyword
            part_match = None
            for part_name, keywords in self.part_keywords.items():
                if word_lower in keywords or word_lower in part_name:
                    part_match = part_name
                    break
            
            if part_match:
                if part_match in detected_parts:
                    matches['matched_parts'].append((start_pos, end_pos, word, part_match))
                else:
                    matches['unmatched_parts'].append((start_pos, end_pos, word, part_match))
                continue
            
            # Check severity
            for severity_level, keywords in self.severity_keywords.items():
                if word_lower in keywords:
                    matches['severity'].append((start_pos, end_pos, word, severity_level))
                    break
            
            # Check damage keywords
            if word_lower in self.damage_keywords:
                matches['damage'].append((start_pos, end_pos, word, 'damage'))
        
        return matches
    
    def _generate_highlighted_html(self, text: str, matches: Dict) -> str:
        """Generate HTML with color-coded highlighting"""
        # Sort all matches by position
        all_matches = []
        
        for match_list, category in [(matches['matched_parts'], 'matched'),
                                       (matches['unmatched_parts'], 'unmatched'),
                                       (matches['severity'], 'severity'),
                                       (matches['damage'], 'damage')]:
            for start, end, word, meta in match_list:
                all_matches.append((start, end, word, category, meta))
        
        all_matches.sort(key=lambda x: x[0])
        
        # Build HTML
        html_parts = []
        last_pos = 0
        
        for start, end, word, category, meta in all_matches:
            # Add text before match
            if start > last_pos:
                html_parts.append(text[last_pos:start])
            
            # Add highlighted word
            if category == 'matched':
                color = '#22c55e'  # Green - detected and mentioned
                tooltip = f'✓ Detected: {meta}'
            elif category == 'unmatched':
                color = '#ef4444'  # Red - mentioned but not detected
                tooltip = f'⚠ Not detected: {meta}'
            elif category == 'severity':
                color = '#f59e0b'  # Orange - severity indicator
                tooltip = f'Severity: {meta}'
            else:  # damage
                color = '#3b82f6'  # Blue - damage action
                tooltip = 'Damage indicator'
            
            html_parts.append(
                f'<span style="background: {color}20; color: {color}; '
                f'padding: 2px 6px; border-radius: 4px; font-weight: 600; '
                f'border-bottom: 2px solid {color};" title="{tooltip}">{word}</span>'
            )
            
            last_pos = end
        
        # Add remaining text
        if last_pos < len(text):
            html_parts.append(text[last_pos:])
        
        return ''.join(html_parts)
    
    def _calculate_statistics(self, matches: Dict, detected_parts: List[str]) -> Dict:
        """Calculate match statistics"""
        total_parts_mentioned = len(matches['matched_parts']) + len(matches['unmatched_parts'])
        matched_count = len(matches['matched_parts'])
        
        match_rate = (matched_count / total_parts_mentioned * 100) if total_parts_mentioned > 0 else 0
        
        return {
            'total_parts_mentioned': total_parts_mentioned,
            'matched_parts': matched_count,
            'unmatched_parts': len(matches['unmatched_parts']),
            'severity_terms': len(matches['severity']),
            'damage_terms': len(matches['damage']),
            'match_rate': round(match_rate, 1),
            'detected_but_not_mentioned': len([p for p in detected_parts 
                                               if not any(p in m[3] for m in matches['matched_parts'])])
        }
    
    def _generate_insights(self, matches: Dict, detected_parts: List[str], text: str) -> List[str]:
        """Generate actionable insights"""
        insights = []
        
        stats = self._calculate_statistics(matches, detected_parts)
        
        # Match rate insight
        if stats['match_rate'] >= 80:
            insights.append("✅ Excellent consistency - description matches detected damage")
        elif stats['match_rate'] >= 50:
            insights.append("⚠ Moderate consistency - some discrepancies found")
        else:
            insights.append("🚨 Low consistency - significant mismatch between description and detection")
        
        # Unmentioned parts
        if stats['detected_but_not_mentioned'] > 0:
            insights.append(f"📝 {stats['detected_but_not_mentioned']} detected part(s) not mentioned in description")
        
        # Unmatched mentions
        if stats['unmatched_parts'] > 0:
            unmatched_names = list(set([m[3] for m in matches['unmatched_parts']]))
            insights.append(f"⚠ Mentioned but not detected: {', '.join(unmatched_names)}")
        
        # Severity assessment
        if stats['severity_terms'] == 0:
            insights.append("ℹ️ No severity terms used in description")
        elif stats['severity_terms'] >= 3:
            insights.append(f"⚠ Multiple severity terms detected ({stats['severity_terms']}) - verify claim accuracy")
        
        # Description quality
        word_count = len(text.split())
        if word_count < 10:
            insights.append("⚠ Brief description - request more details for better assessment")
        elif word_count > 100:
            insights.append("ℹ️ Detailed description provided - good for analysis")
        
        return insights
