from abc import ABC
from ignite.engine import Engine, EventEnum


class Handler:

    name: str
    events: set[EventEnum]
    priorities: tuple[int, ...]

    def __init__(self, name: str, events: set[EventEnum], priorities: tuple[int, ...]):
        self.name = name
        self.events = events
        self.priorities = priorities


class HandlerManager:

    def __init__(self, events):

        self._handlers: dict[str, Handler] = {}
        self._registered: set = set()
        self._events = events

    @property
    def handlers(self):
        return self._handlers

    def add_handler(self, handler: Handler):
        self._handlers[handler.name] = handler

    def get_handler(self, name: str) -> Handler:
        if name not in self._handlers:
            raise ValueError(f"Handler {name} not found.")
        return self._handlers[name]

    def get_handlers_by_event(self, sort=True) -> dict[EventEnum, list[Handler]]:
        events_handlers = {event: [] for event in self._events}
        for handler in self._handlers.values():
            for event in handler.events:
                events_handlers[event].append(handler)
        if sort:
            for event, handlers in events_handlers.items():
                handlers.sort(key=lambda x: x.priorities)
        return events_handlers

    def register_to_engine(self, engine: Engine):

        events_handlers = self.get_handlers_by_event()
        to_be_registered = set()
        for event_enum, handlers in events_handlers.items():
            for handler in handlers:
                if handler.name in self._registered:
                    continue
                handler_event = next(
                    (e for e in handler.events if e == event_enum), None
                )  # event_enum is defined in EventEnum
                # it's without every info
                engine.add_event_handler(
                    handler_event, handler.get_event_handler(event_enum)
                )
                to_be_registered.add(handler.name)
        self._registered.update(to_be_registered)


class MolpotEngine(ABC):

    events: EventEnum

    def __init__(self):

        self._engines: dict[str, Engine] = {}
        self._handlers = HandlerManager(self.__class__.events)

    def add_engine(self, name: str, engine: Engine):

        self._engines[name] = engine
        setattr(self, name, engine)

    def add_handler(self, handler: Handler):

        self._handlers.add_handler(handler)

    def get_handler(self, name: str) -> Handler:
        return self._handlers.get_handler(name)