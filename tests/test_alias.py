import molpot as mpot
import pytest

class TestAlias:

    @pytest.fixture(name="alias", scope="class")
    def test_alias(self):
        return mpot.Alias("test", int, "unit", "comment")
    
    @pytest.fixture(name="namespace", scope="class")
    def test_namespace(self):
        return mpot.NameSpace("Test")
    
    def test_in_namespace(self, alias, namespace):

        expected_alias1 = namespace[alias]
        expected_alias2 = namespace[alias.name]

        assert expected_alias1 == "test"
        assert expected_alias2 == "test"

    def test_global_get(self):

        ns = mpot.NameSpace("Test")
        assert ns.name == "Test"

        alias = ns["test"]
        assert alias.name == "test"

        mpot.Config()