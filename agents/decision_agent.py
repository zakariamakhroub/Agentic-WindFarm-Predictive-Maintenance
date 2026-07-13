class DecisionAgent:

    def make_decision(self, summary):

        probability = summary["max_fault_probability"]

        if probability >= 0.85:
            risk = "Critical"
            action = "Immediate Maintenance"
            priority = 4

        elif probability >= 0.60:
            risk = "High"
            action = "Schedule Maintenance"
            priority = 3

        elif probability >= 0.30:
            risk = "Medium"
            action = "Increase Monitoring"
            priority = 2

        else:
            risk = "Low"
            action = "Continue Operation"
            priority = 1

        summary["risk_level"] = risk
        summary["recommended_action"] = action
        summary["priority_score"] = priority

        return summary