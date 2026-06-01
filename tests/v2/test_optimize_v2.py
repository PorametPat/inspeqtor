import inspeqtor as sq


def test_edge_scheduler():
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]
    result = sq.optimize.edge_scheduler(edges)

    assert len(result) == 2

    assert result[0] == [(0, 1), (2, 3), (4, 5)]
    assert result[1] == [(1, 2), (3, 4), (5, 0)]


def test_extract_sub_distribution():
    empirical_distribution = {(0, 0): 11, (0, 1): 12, (1, 0): 13, (1, 1): 14}
    result = sq.optimize.extract_sub_distribution(empirical_distribution, [(0,), (1,)])

    assert result == {
        (0,): {(0,): 11 + 12, (1,): 13 + 14},
        (1,): {(0,): 11 + 13, (1,): 12 + 14},
    }
