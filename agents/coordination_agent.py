class CoordinationAgent:

    def prioritize(self, decisions):

        ranked = sorted(
            decisions,
            key=lambda x: (
                x["priority_score"],
                x["max_fault_probability"]
            ),
            reverse=True
        )

        for i, decision in enumerate(ranked, start=1):
            decision["rank"] = i

        return ranked