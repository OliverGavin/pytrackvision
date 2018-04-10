from abc import ABC, abstractmethod
from collections import deque

from .gesture import Gesture


class GestureState(ABC):

    @abstractmethod
    def run(self):
        ...

    @abstractmethod
    def next(self, gesture):
        ...


class GestureStateTransition(GestureState):

    def __init__(self):
        self.transitions = None

    def next(self, gesture):
        if gesture in self.transitions:
            return self.transitions[gesture]
        else:
            raise ValueError(f"Gesture transition ({gesture.name}) not supported from the current state")


class HandPending(GestureStateTransition):

    def run(self):
        # print("Waiting for an opened hand")
        return Gesture.NONE

    def next(self, gesture):
        if not self.transitions:
            self.transitions = {
                Gesture.NONE: GestureStateMachine.hand_pending,
                Gesture.HAND_OPENED: GestureStateMachine.hand_opened
            }
        return GestureStateTransition.next(self, gesture)


class HandOpened(GestureStateTransition):

    def run(self):
        # print("Opened hand")
        return Gesture.HAND_OPENED

    def next(self, gesture):
        if not self.transitions:
            self.transitions = {
                Gesture.HAND_OPENED: GestureStateMachine.hand_opened,
                Gesture.NONE: GestureStateMachine.hand_pending,
                Gesture.HAND_CLOSED: GestureStateMachine.hand_closed,
                Gesture.HAND: GestureStateMachine.hand_unkown
            }
        return GestureStateTransition.next(self, gesture)


class HandClosed(GestureStateTransition):

    def run(self):
        # print("Closed hand")
        return Gesture.HAND_CLOSED

    def next(self, gesture):
        if not self.transitions:
            self.transitions = {
                Gesture.HAND_CLOSED: GestureStateMachine.hand_closed,
                Gesture.HAND_OPENED: GestureStateMachine.hand_opened,
                Gesture.NONE: GestureStateMachine.hand_pending,
                Gesture.HAND: GestureStateMachine.hand_unkown
            }
        return GestureStateTransition.next(self, gesture)


class HandUnkown(GestureStateTransition):

    def run(self):
        # print("Unknown hand")
        return Gesture.HAND

    def next(self, gesture):
        if not self.transitions:
            self.transitions = {
                Gesture.HAND: GestureStateMachine.hand_unkown,
                Gesture.HAND_CLOSED: GestureStateMachine.hand_closed,
                Gesture.HAND_OPENED: GestureStateMachine.hand_opened,
                Gesture.NONE: GestureStateMachine.hand_pending
            }
        return GestureStateTransition.next(self, gesture)


class GestureStateMachine:
    """
    Simple state machine allowing transitions in certain conditions and reducing
    rapid changes between gestures.
    """

    hand_pending = HandPending()
    hand_opened = HandOpened()
    hand_closed = HandClosed()
    hand_unkown = HandUnkown()

    def __init__(self, initialState=None):
        self.current_state = initialState if initialState else GestureStateMachine.hand_pending
        self.current_state.run()
        self.incomming_gestures = deque(maxlen=3)

    def run(self, gesture):
        # print('input: ', gesture)
        try:
            new_state = self.current_state.next(gesture)
        except ValueError:
            new_state = self.current_state

        self.incomming_gestures.append(new_state.run())
        if all([g == gesture for g in self.incomming_gestures]):
            self.current_state = new_state

        return self.current_state.run()
