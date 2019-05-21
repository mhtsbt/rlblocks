import unittest

from rlblocks.memory.prioritized_experience_replay import PrioritizedExperienceReplayMemory


class TestPrioritizedExperienceReplayMemory(unittest.TestCase):

    def setUp(self):
        self.memory = PrioritizedExperienceReplayMemory()


class TestStore(TestPrioritizedExperienceReplayMemory):

    def test_store_and_retreive(self):

        self.assertEqual(len(self.memory.data), 0)

        test_transition = (1, 2, 3)
        self.memory.store(test_transition)

        self.assertEqual(len(self.memory.data), 1)

        sample = self.memory.sample(1)
        self.assertTrue(sample.__contains__(test_transition))

