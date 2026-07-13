class DecisionAgent:

    def make_decision(self, summary):

        mean_probability = summary["mean_fault_probability"]
        max_probability = summary["max_fault_probability"]

        # Combined daily priority score: balance overall daily behaviour with severe spikes
        priority_score = 0.7 * mean_probability + 0.3 * max_probability

        if priority_score >= 0.75 and mean_probability >= 0.55:
            risk = "Critical"
            action = "Immediate Maintenance"
            priority = 4

        elif priority_score >= 0.45 and mean_probability >= 0.20:
            risk = "High"
            action = "Schedule Maintenance"
            priority = 3

        elif priority_score >= 0.20:
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
        summary["daily_priority_score"] = round(priority_score, 4)

        return summary