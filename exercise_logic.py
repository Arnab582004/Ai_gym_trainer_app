class ExerciseCounter:
    def __init__(self):
        self.count = 0
        self.stage = None   # "up" or "down"

    def pushup_counter(self, angle):
        # angle reference:
        # down position → small angle (~50-70)
        # up position → larger angle (~160-170)

        if angle > 150:
            self.stage = "up"

        if angle < 70 and self.stage == "up":
            self.stage = "down"
            self.count += 1

        return self.count, self.stage
