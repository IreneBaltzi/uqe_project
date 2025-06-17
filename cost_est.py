class CostEstimator:
    def __init__(self):
        self.costs = []

    def add(self, count, label):
        self.costs.append((label, count))

    def report(self):
        report_lines = ["[COST REPORT]"]
        total = 0
        for label, count in self.costs:
            report_lines.append(f"  {label}: {count} LLM calls")
            total += count
        report_lines.append(f"  Total LLM calls: {total}")
        summary = "\n".join(report_lines)
        # print(summary)
        return summary  

