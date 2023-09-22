from typing import Hashable


class Borg:
    """
    Base implementation of the borg idiom.

    This is used for objects that must have a unique shared state.
    """
    _borg_registry = {}

    def __init__(self):
        super().__init__()
        # Ensure that there is one shared state
        Borg._borg_registry.setdefault(self.borg_state_id, self)
        # Ensure that this instance shares the state
        drone = Borg._borg_registry[self.borg_state_id]
        assert drone.borg_state_id == self.borg_state_id, "Attempt to create borg drones with different state_id"
        assert type(drone) == type(
            self), f"Attempt to create borg drones with different types {type(drone)} != {type(self)}"
        self.__dict__ = drone.__dict__

    @property
    def borg_state_id(self) -> Hashable:
        raise NotImplementedError("Must be implemented to distinguish different states")
