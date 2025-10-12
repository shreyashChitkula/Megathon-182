# ✅ Severity Levels - Clarification

## Current System: 5 Severity Levels

**ESPER uses a rule-based severity assessment with 5 distinct levels:**

```
MINIMAL (0)    → No damage or very minor
MINOR (1)      → Small cosmetic damage  
MODERATE (2)   → Noticeable damage requiring repair
SEVERE (3)     → Major damage, significant repair
TOTAL_LOSS (4) → Vehicle may be write-off
```

---

## Score Mapping

| Severity Level | Score Range | Description | Examples |
|---------------|-------------|-------------|----------|
| **MINIMAL** | < 0.15 | No damage or very minor | Surface scratches, tiny dings |
| **MINOR** | 0.15 - 0.35 | Cosmetic, easy repair | Small dent, paint chip |
| **MODERATE** | 0.35 - 0.60 | Professional repair needed | Multiple dents, cracked light |
| **SEVERE** | 0.60 - 0.85 | Major damage, high cost | Windscreen damage, structural |
| **TOTAL_LOSS** | > 0.85 | Vehicle may be write-off | Multiple critical damages |

---

## Why 5 Levels (Not 3)?

**More Granular Assessment:**
- Better reflects reality (not everything is MINOR/MODERATE/SEVERE)
- MINIMAL handles clean vehicles / very tiny damage
- TOTAL_LOSS identifies write-off candidates

**Business Value:**
- MINIMAL → Fast-track approval
- MINOR → Simple repair process
- MODERATE → Standard assessment
- SEVERE → Detailed investigation
- TOTAL_LOSS → Write-off consideration

---

## Implementation

**File:** `rule_based_severity.py` lines 23-28

```python
class SeverityLevel(IntEnum):
    """Severity levels matching your existing system."""
    MINIMAL = 0    # No damage or very minor
    MINOR = 1      # Small cosmetic damage
    MODERATE = 2   # Noticeable damage requiring repair
    SEVERE = 3     # Major damage, significant repair
    TOTAL_LOSS = 4 # Vehicle may be write-off
```

**Active in:** `app.py` line 458-501

---

## Examples from Real Claims

### MINIMAL (Score: 0.12)
```
Detections: 1 damage (RunningBoard-Dent)
Area: 2%
Explanation: "Single minor damage detected. Minimal repair needed."
```

### MINOR (Score: 0.28)
```
Detections: 2 damages (front-bumper-dent, small scratch)
Area: 5%
Explanation: "2 cosmetic damages detected. Minor repair recommended."
```

### MODERATE (Score: 0.52)
```
Detections: 3 damages (doorouter-dent, fender-dent, Headlight-Damage)
Area: 12%
Explanation: "3 damages detected including functional parts. 
Professional repair required."
```

### SEVERE (Score: 0.72)
```
Detections: 4 damages (Front-Windscreen-Damage, bonnet-dent, pillar-dent)
Area: 18%
Critical: Yes
Explanation: "Critical damage detected (Front-Windscreen). 
Structural concerns. Severe damage requiring comprehensive repair."
```

### TOTAL_LOSS (Score: 0.92)
```
Detections: 8+ damages (multiple critical)
Area: 35%
Critical: Yes
Explanation: "Extensive damage across multiple critical areas. 
Vehicle integrity compromised. Total loss consideration recommended."
```

---

## Special Rules

**Overrides normal score mapping:**

1. **Critical Damage Present** → Minimum MODERATE
   - Front/Rear Windscreen (weight ≥ 0.90)
   - Pillar damage (weight ≥ 0.85)

2. **High Damage Count** → Escalates severity
   - 5+ damages → Minimum MODERATE
   - 7+ damages → SEVERE

3. **Large Damage Area** → Boosts score
   - > 25% image area → Score +0.2

---

## UI Display

**In estimate.html:**
```html
<!-- Badge shows level -->
<span class="severity-badge severity-MODERATE">
    MODERATE
</span>

<!-- Explainable section shows details -->
💡 Explainable Severity Assessment
3 damages detected. Significant damage area (12.4%). 
Moderate damage requiring professional repair.

Factors:
- Area Score: 65%
- Count Score: 36%
- Type Score: 81%
```

---

## Comparison with Old System

| Old (CNN-based) | New (Rule-based) |
|-----------------|------------------|
| 3 classes (MINOR/MODERATE/SEVERE) | 5 levels (MINIMAL to TOTAL_LOSS) |
| Black box | Transparent |
| No MINIMAL or TOTAL_LOSS | Covers full spectrum |
| Fixed classes | Granular levels |

---

## For Presentation

**When explaining severity:**

"Our system doesn't just say MINOR or SEVERE. It provides **5 granular levels** from MINIMAL to TOTAL_LOSS, each with clear score ranges and transparent calculation.

This gives insurers better decision-making:
- **MINIMAL** → Instant approval
- **MINOR** → Fast-track
- **MODERATE** → Standard process
- **SEVERE** → Detailed review
- **TOTAL_LOSS** → Write-off assessment

Every level comes with full explanation of why it was assigned."

---

## Verification

**To confirm 5 levels are active:**

```bash
# Check rule_based_severity.py
grep -A 5 "class SeverityLevel" rule_based_severity.py

# Check it's being used in app.py
grep "rule_based_severity" app.py

# Check console output shows all 5 levels
# Run app and upload images - should see levels like:
# ✓ Rule-Based Severity: MODERATE (Score: 0.593, Confidence: 91.02%)
```

---

## Summary

✅ **System Status:** Rule-based severity with **5 levels** is **ACTIVE**  
✅ **Documentation:** Updated to reflect 5 levels (not 3)  
✅ **Code:** `rule_based_severity.py` defines all 5 levels  
✅ **Integration:** `app.py` uses rule-based as primary method  
✅ **UI:** `estimate.html` displays all 5 levels correctly  

**All documentation now consistent with 5-level system!**
