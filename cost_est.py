class CostEstimator:
    def __init__(self):
        self.cost = 0
        self.notes = []

    def add(self, count, note):
        self.cost += count
        self.notes.append((note, count))

    def report(self):
        print("\n[ESTIMATED COST]")
        for note, count in self.notes:
            print(f"  {note}: {count} LLM calls")
        print(f"  Total: {self.cost} LLM calls\n")
