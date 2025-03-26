import threading
import abc


def synchronized(method):
    def f(*args):
        self = args[0]
        self.mutex.acquire()
        # print(method.__name__, 'acquired')
        try:
            return method(args)
        finally:
            self.mutex.release()
            # print(method.__name__, 'released')
    return f


def synchronize(klass, names=None):
    """Synchronize methods in the given class.
    Only synchronize the methods whose names are
    given, or all methods if names=None."""
    if type(names)==type(''): names = names.split()
    for (name, val) in klass.__dict__.items():
        if callable(val) and name != '__init__' and \
          (names == None or name in names):
            # print("synchronizing", name)
            klass.__dict__[name] = synchronized(val)


class Synchronization:
    def __init__(self):
        self.mutex = threading.RLock()


class Observer(object):

    @abc.abstractmethod
    def update(self, observable, arg):
        """
        Called when the observed object is
        modified. You call an Observable object's
        notifyObservers method to notify all the
        object's observers of the change
        :param observable:
        :param arg:
        :return:
        """
        pass


class Observable(Synchronization):
    def __init__(self):
        self.obs = []
        self.changed = 0
        Synchronization.__init__(self)

    def add_observer(self, observer):
        if observer not in self.obs:
            self.obs.append(observer)

    def delete_observer(self, observer):
        self.obs.remove(observer)

    def notify_observers(self, arg=None):
        """
        If 'changed' indicates that this object
        has changed, notify all its observers, then
        call clearChanged(). Each observer has its
        update() called with two arguments: this
        observable object and the generic 'arg'.
        :param arg:
        :return:
        """

        self.mutex.acquire()
        try:
            if not self.changed: return
            # Make a local copy in case of synchronous
            # additions of observers:
            localArray = self.obs[:]
            self.clear_changed()
        finally:
            self.mutex.release()
        # Updating is not required to be synchronized:
        for observer in localArray:
            observer.update(self, arg)

    def delete_observers(self): self.obs = []

    def set_changed(self): self.changed = 1

    def clear_changed(self): self.changed = 0

    def has_changed(self): return self.changed

    def count_observers(self): return len(self.obs)
