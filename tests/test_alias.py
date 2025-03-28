import molpot as mpot
import pytest

class TestAlias:

    @pytest.fixture(name="alias", scope="class")
    def test_alias(self):
        return mpot.Alias("test", "comment", str, "unit", (), "Namespace")
    
    @pytest.fixture(name="namespace", scope="class")
    def test_namespace(self, alias):
        ns = mpot.NameSpace("Namespace")
        ns.add(alias)
        return ns
    
    def test_alias_eq(self, alias):
        assert alias == ("Namespace", "test")
        assert alias == alias
    
    def test_in_namespace(self, alias, namespace):
        expected_alias1 = namespace[alias]
        expected_alias2 = namespace[alias.name]
        assert expected_alias1 == ("Namespace", "test")
        assert expected_alias2 == ("Namespace", "test")

    def test_global_get(self):

        ns = mpot.NameSpace("Namespace")
        assert ns.name == "Namespace"

        alias = ns["test"]
        assert alias.name == "test"
