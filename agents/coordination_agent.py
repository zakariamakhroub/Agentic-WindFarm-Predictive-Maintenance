class CoordinationAgent:

    def prioritize(self, decisions):

        ranked = sorted(
            decisions,
            key=lambda x: (
                x.get("daily_priority_score", x.get("priority_score", 0)),
                x.get("max_fault_probability", 0),
                x.get("mean_fault_probability", 0)
            ),
            reverse=True
        )

        for i, decision in enumerate(ranked, start=1):
            decision["rank"] = i

        return ranked
    
