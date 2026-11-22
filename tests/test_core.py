import random
import unittest
from indecisive.core import Signal, Task, choose, evaluate, update, response_probability

class TestExactE0(unittest.TestCase):
    def test_bayes(self):
        s = Signal('q', .8, .2, .1)
        self.assertAlmostEqual(update(.5, s, True), .8)
        self.assertAlmostEqual(update(.5, s, False), .2)
        self.assertAlmostEqual(sum(response_probability(.3, s, v) for v in (False, True)), 1)
        self.assertAlmostEqual(sum(response_probability(.3, s, v) * update(.3, s, v)
                                   for v in (False, True)), .3)

    def test_impossible_observation(self):
        with self.assertRaises(ValueError):
            update(0.0, Signal('q', 1, 0, 0), True)

    def test_voi_numeric(self):
        t = Task(.5, (Signal('q', .8, .2, .1),), max_queries=1)
        self.assertEqual(choose(t, .5, (0,), 1)[0], ('query', 0))
        self.assertAlmostEqual(evaluate(t), .5)
        self.assertAlmostEqual(evaluate(t, 'greedy'), 0)

    def test_skip_uninformative_or_costly(self):
        for s in (Signal('noise', .5, .5, .01), Signal('costly', .8, .2, 2)):
            t = Task(.5, (s,))
            self.assertEqual(choose(t, .5, (0,), 3)[0][0], 'commit')

    def test_unbounded_upper(self):
        t = Task(.35, (Signal('q', .9, .1, .04),))
        self.assertLessEqual(evaluate(t), 1.0)

    def test_exact_weakly_dominates_greedy_and_rolling_one_step(self):
        r = random.Random(2042)
        for _ in range(100):
            ss = tuple(Signal(str(i), r.uniform(.53, .98), r.uniform(.02, .47),
                              r.uniform(.001, .24)) for i in range(3))
            t = Task(r.uniform(.1, .9), ss, gamma=r.uniform(.86, 1), max_queries=3)
            optimum = evaluate(t)
            self.assertGreaterEqual(optimum + 1e-9, evaluate(t, 'one_step'))
            self.assertGreaterEqual(optimum + 1e-9, evaluate(t, 'greedy'))
            self.assertAlmostEqual(optimum,
                 choose(t, t.prior, tuple(range(len(ss))), t.max_queries)[1], places=9)

    def test_trace_preaction(self):
        t = Task(.5, (Signal('q', .8, .2, .01),))
        val, events = evaluate(t, trace=True)
        self.assertGreaterEqual(len(events), 3)
        self.assertTrue(all(event['forecast_frozen_before_action'] for event in events))
        self.assertAlmostEqual(val, .59)

if __name__ == '__main__':
    unittest.main()
