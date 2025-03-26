# implementation motivated from: https://refactoring.guru/design-patterns/observer/python/example

from abc import ABC, abstractmethod
from typing import List
from utils.util.enum_conversion import StringEnum


class Observer(ABC):
    """
    The Observer interface declares the update method, used by subjects.
    """

    @abstractmethod
    def update(self, event:StringEnum) -> None:
        """
        Get update from observable.
        """
        pass


class ConcreteObserver(Observer):
    def __init__(self, observables_events:[()]):
        self.observables_events=observables_events
        observer = self

        #for each observable-event-tuple do:
        for observable, event in observables_events:
            observable.attach(observer, event)

    def update(self, event:StringEnum) -> None:
        pass


class AbstractObservable(ABC):
    """
    Abstract class for defining methods interacting with observers
    """
    def __init__(self):
        self._observers=None

    @abstractmethod
    def attach(self, observer: Observer) -> None:
        """
        Attach an observer to the observable.
        """
        pass

    @abstractmethod
    def detach(self, observer: Observer) -> None:
        """
        Detach an observer from the observable.
        """
        pass

    @abstractmethod
    def notify_observers(self, event) -> None:
        """
        Notify all observers about an event.
        """
        pass


class ConcreteObservable(AbstractObservable):
    """
    Concrete observable implementation class for defining methods interacting with observers
    """

    def __init__(self, events:[StringEnum]):
        super().__init__()
        self._observers :{StringEnum:[Observer]} = {}
        for event in events:
            self._observers[event.value]=[]


    def attach(self, observer: Observer, event:StringEnum) -> None:
        if not observer in self._observers[event.value]:
            self._observers[event.value].append(observer)

    def detach(self, observer: Observer, event:StringEnum) -> None:
        if observer in self._observers[event.value]:
            self._observers[event.value].remove(observer)

    def notify_observers(self, event) -> None:
        """
        Update each observer which is registered to a specific event
        """

        interested_observers=self._observers[event.value]
        for observer in interested_observers:
            observer.update(event)


