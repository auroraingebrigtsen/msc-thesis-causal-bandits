from .base import BaseSchedule

class StationarySchedule(BaseSchedule):
    def next(self, t: int) -> None:
        return None
    
    def get_change_points(self, T:int) -> list[int]:
        return []
    
    def reset(self) -> None:
        pass