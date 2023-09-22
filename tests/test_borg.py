import pytest

from pycheribenchplot.core.borg import Borg


@pytest.fixture
def mock_borg(mocker):
    fake_borg_registry = {}
    mocker.patch.object(Borg, "_borg_registry", new=fake_borg_registry)

    return fake_borg_registry


def test_simple_borg(mock_borg):
    class My(Borg):
        def __init__(self, name):
            # Must be visible in Borg.__init__()
            self.name = name
            super().__init__()
            self.value = 10

        @property
        def borg_state_id(self):
            return self.name

    foo = My("foo")
    foo2 = My("foo")
    bar = My("bar")

    assert "foo" in Borg._borg_registry
    assert "bar" in Borg._borg_registry

    assert foo.__dict__ is foo2.__dict__

    foo.value = 20
    assert foo2.value == 20
    assert bar.value == 10

    bar.value = 30
    assert foo.value == 20
    assert foo2.value == 20
    assert bar.value == 30


def test_borg_inheritance(mock_borg):
    class Parent(Borg):
        def __init__(self, name):
            self.name = name
            super().__init__()

        @property
        def borg_state_id(self):
            return self.name

    class Child(Parent):
        def __init__(self, name):
            super().__init__(name)
            self.child_value = 10

    c1 = Child("foo")
    c2 = Child("foo")
    c3 = Child("bar")

    assert c1.__dict__ is c2.__dict__

    c1.child_value = 20
    assert c1.child_value == 20
    assert c2.child_value == 20
    assert c3.child_value == 10

    c3.child_value = 30
    assert c1.child_value == 20
    assert c2.child_value == 20
    assert c3.child_value == 30


def test_borg_with_metaclass(mock_borg):
    class Meta(type):
        def __new__(mcls, name, bases, ns):
            ns["shared_state"] = {}
            return super().__new__(mcls, name, bases, ns)

    class My(Borg, metaclass=Meta):
        def __init__(self, name):
            self.name = name
            super().__init__()

        @property
        def borg_state_id(self):
            return self.name

    class Other(metaclass=Meta):
        pass

    a = My("foo")
    b = My("foo")
    c = My("bar")
    o = Other()

    a.shared_state["v"] = 10
    assert b.shared_state["v"] == 10
    assert c.shared_state["v"] == 10
    assert len(o.shared_state) == 0

    assert a.__dict__ is b.__dict__
    assert a.__dict__ is not c.__dict__
