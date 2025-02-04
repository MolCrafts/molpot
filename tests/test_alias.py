import molpot as mpot
import pytest

class TestAlias:

    @pytest.fixture(name="alias", scope="class")
    def test_alias(self):
        return mpot.Alias("test", "comment", str, "unit", (), "category")
    
    @pytest.fixture(name="namespace", scope="class")
    def test_namespace(self, alias):
        ns = mpot.NameSpace("Test")
        ns.add(alias)
        return ns
    
    def test_alias_eq(self, alias):
        assert alias == ("category", "test")
        assert alias == alias
    
    def test_in_namespace(self, alias, namespace):
        expected_alias1 = namespace[alias]
        expected_alias2 = namespace[alias.name]
        print(expected_alias1.key)
        assert expected_alias1 == ("Test", "category", "test")
        assert expected_alias2 == ("Test", "category", "test")

    def test_global_get(self):

        ns = mpot.NameSpace("Test")
        assert ns.name == "Test"

        alias = ns["test"]
        assert alias.name == "test"
