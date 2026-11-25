# Puzzle Dump Recorded Scenario Conversion Report

- source_kind: puzzle_dump_current_turn_fixture
- input_dir: puzzle_dump
- output_dir: decision_engine/tests/fixtures/recorded_scenarios/puzzle_dump
- total_input_files: 56
- written_scenarios: 49
- adapter_gap: 0
- needs_root_patch: 7
- executable transitions: 0
- note: These are independent current-turn puzzle starts, not action replays.

## Aggregate action counts

- play: 139
- attack_leader: 8
- attack_follower: 51
- evolve: 10
- super_evolve: 6
- activate: 7
- fusion: 13
- special_action: 0
- extra_pp: 0
- turn_end: 49
- turn_start: 0

## Items

| input | scenario_id | status | actions | non_turn_end | play | attack_leader | attack_follower | evolve | super_evolve | reasons |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| puzzle_dump/puzzle_0001_20260701_151548_536.json | puzzle_dump_puzzle_0001_20260701_151548_536 | needs_root_patch | 0 | 0 | 0 | 0 | 0 | 0 | 0 | is_turn_ended is not false; no non-turn-end protocol action is enumerable |
| puzzle_dump/puzzle_0002_20260701_151548_545.json | puzzle_dump_puzzle_0002_20260701_151548_545 | needs_root_patch | 1 | 0 | 0 | 0 | 0 | 0 | 0 | no non-turn-end protocol action is enumerable |
| puzzle_dump/puzzle_0003_20260701_151559_576.json | puzzle_dump_puzzle_0003_20260701_151559_576 | needs_root_patch | 0 | 0 | 0 | 0 | 0 | 0 | 0 | is_turn_ended is not false; no non-turn-end protocol action is enumerable |
| puzzle_dump/puzzle_0004_20260701_151559_583.json | puzzle_dump_puzzle_0004_20260701_151559_583 | needs_root_patch | 0 | 0 | 0 | 0 | 0 | 0 | 0 | is_ally_turn is not true; no non-turn-end protocol action is enumerable |
| puzzle_dump/puzzle_0005_20260701_151559_616.json | puzzle_dump_puzzle_0005_20260701_151559_616 | needs_root_patch | 0 | 0 | 0 | 0 | 0 | 0 | 0 | is_ally_turn is not true; no non-turn-end protocol action is enumerable |
| puzzle_dump/puzzle_0006_20260701_151559_658.json | puzzle_dump_puzzle_0006_20260701_151559_658 | needs_root_patch | 0 | 0 | 0 | 0 | 0 | 0 | 0 | is_ally_turn is not true; no non-turn-end protocol action is enumerable |
| puzzle_dump/puzzle_0007_20260701_151559_673.json | puzzle_dump_puzzle_0007_20260701_151559_673 | needs_root_patch | 0 | 0 | 0 | 0 | 0 | 0 | 0 | is_ally_turn is not true; is_turn_ended is not false; no non-turn-end protocol action is enumerable |
| puzzle_dump/puzzle_0008_20260701_151559_678.json | puzzle_dump_puzzle_0008_20260701_151559_678 | written | 4 | 3 | 3 | 0 | 0 | 0 | 0 |  |
| puzzle_dump/puzzle_0009_20260701_151644_421.json | puzzle_dump_puzzle_0009_20260701_151644_421 | written | 4 | 3 | 3 | 0 | 0 | 0 | 0 |  |
| puzzle_dump/state_0001_20260701_163140_583.json | puzzle_dump_state_0001_20260701_163140_583 | written_with_root_patch | 5 | 4 | 3 | 0 | 0 | 1 | 0 |  |
| puzzle_dump/state_0001_20260701_174511_925.json | puzzle_dump_state_0001_20260701_174511_925 | written_with_root_patch | 9 | 8 | 7 | 0 | 0 | 1 | 0 |  |
| puzzle_dump/state_0001_20260701_175144_507.json | puzzle_dump_state_0001_20260701_175144_507 | written_with_root_patch | 10 | 9 | 6 | 0 | 0 | 0 | 1 |  |
| puzzle_dump/state_0001_20260702_120632_482.json | puzzle_dump_state_0001_20260702_120632_482 | written_with_root_patch | 5 | 4 | 4 | 0 | 0 | 0 | 0 |  |
| puzzle_dump/state_0001_20260702_123843_342.json | puzzle_dump_state_0001_20260702_123843_342 | written | 3 | 2 | 2 | 0 | 0 | 0 | 0 |  |
| puzzle_dump/state_0001_20260702_124434_506.json | puzzle_dump_state_0001_20260702_124434_506 | written | 4 | 3 | 3 | 0 | 0 | 0 | 0 |  |
| puzzle_dump/state_0001_20260702_124948_873.json | puzzle_dump_state_0001_20260702_124948_873 | written | 5 | 4 | 3 | 0 | 0 | 0 | 0 |  |
| puzzle_dump/state_0002_20260701_163153_822.json | puzzle_dump_state_0002_20260701_163153_822 | written | 7 | 6 | 4 | 0 | 1 | 1 | 0 |  |
| puzzle_dump/state_0002_20260702_120737_515.json | puzzle_dump_state_0002_20260702_120737_515 | written | 5 | 4 | 4 | 0 | 0 | 0 | 0 |  |
| puzzle_dump/state_0002_20260702_123907_530.json | puzzle_dump_state_0002_20260702_123907_530 | written | 4 | 3 | 3 | 0 | 0 | 0 | 0 |  |
| puzzle_dump/state_0002_20260702_124453_479.json | puzzle_dump_state_0002_20260702_124453_479 | written | 7 | 6 | 2 | 1 | 1 | 1 | 0 |  |
| puzzle_dump/state_0002_20260702_125009_240.json | puzzle_dump_state_0002_20260702_125009_240 | written | 5 | 4 | 2 | 0 | 0 | 0 | 0 |  |
| puzzle_dump/state_0003_20260701_163118_786.json | puzzle_dump_state_0003_20260701_163118_786 | written | 13 | 12 | 2 | 0 | 10 | 0 | 0 |  |
| puzzle_dump/state_0003_20260701_163208_475.json | puzzle_dump_state_0003_20260701_163208_475 | written | 9 | 8 | 4 | 1 | 2 | 0 | 1 |  |
| puzzle_dump/state_0003_20260702_120757_167.json | puzzle_dump_state_0003_20260702_120757_167 | written | 3 | 2 | 1 | 0 | 0 | 0 | 0 |  |
| puzzle_dump/state_0003_20260702_123925_311.json | puzzle_dump_state_0003_20260702_123925_311 | written | 8 | 7 | 2 | 2 | 2 | 0 | 0 |  |
| puzzle_dump/state_0003_20260702_124510_369.json | puzzle_dump_state_0003_20260702_124510_369 | written | 9 | 8 | 2 | 0 | 4 | 0 | 2 |  |
| puzzle_dump/state_0003_20260702_125025_190.json | puzzle_dump_state_0003_20260702_125025_190 | written | 3 | 2 | 2 | 0 | 0 | 0 | 0 |  |
| puzzle_dump/state_0004_20260701_163224_387.json | puzzle_dump_state_0004_20260701_163224_387 | written | 14 | 13 | 7 | 0 | 5 | 1 | 0 |  |
| puzzle_dump/state_0004_20260702_120816_954.json | puzzle_dump_state_0004_20260702_120816_954 | written | 5 | 4 | 4 | 0 | 0 | 0 | 0 |  |
| puzzle_dump/state_0004_20260702_123945_452.json | puzzle_dump_state_0004_20260702_123945_452 | written | 4 | 3 | 3 | 0 | 0 | 0 | 0 |  |
| puzzle_dump/state_0004_20260702_124529_788.json | puzzle_dump_state_0004_20260702_124529_788 | written | 5 | 4 | 4 | 0 | 0 | 0 | 0 |  |
| puzzle_dump/state_0004_20260702_125041_261.json | puzzle_dump_state_0004_20260702_125041_261 | written | 6 | 5 | 1 | 0 | 3 | 0 | 1 |  |
| puzzle_dump/state_0005_20260701_163304_147.json | puzzle_dump_state_0005_20260701_163304_147 | written | 12 | 11 | 6 | 1 | 1 | 0 | 1 |  |
| puzzle_dump/state_0005_20260702_120833_031.json | puzzle_dump_state_0005_20260702_120833_031 | written | 12 | 11 | 2 | 0 | 6 | 3 | 0 |  |
| puzzle_dump/state_0005_20260702_124006_573.json | puzzle_dump_state_0005_20260702_124006_573 | written | 5 | 4 | 4 | 0 | 0 | 0 | 0 |  |
| puzzle_dump/state_0005_20260702_124545_359.json | puzzle_dump_state_0005_20260702_124545_359 | written | 4 | 3 | 3 | 0 | 0 | 0 | 0 |  |
| puzzle_dump/state_0005_20260702_125055_945.json | puzzle_dump_state_0005_20260702_125055_945 | written | 4 | 3 | 3 | 0 | 0 | 0 | 0 |  |
| puzzle_dump/state_0006_20260702_124028_256.json | puzzle_dump_state_0006_20260702_124028_256 | written | 7 | 6 | 1 | 1 | 4 | 0 | 0 |  |
| puzzle_dump/state_0006_20260702_124605_167.json | puzzle_dump_state_0006_20260702_124605_167 | written | 8 | 7 | 2 | 1 | 3 | 1 | 0 |  |
| puzzle_dump/state_0007_20260702_124046_307.json | puzzle_dump_state_0007_20260702_124046_307 | written | 4 | 3 | 3 | 0 | 0 | 0 | 0 |  |
| puzzle_dump/state_0007_20260702_124623_206.json | puzzle_dump_state_0007_20260702_124623_206 | written | 8 | 7 | 2 | 0 | 5 | 0 | 0 |  |
| puzzle_dump/state_0008_20260702_124116_754.json | puzzle_dump_state_0008_20260702_124116_754 | written | 4 | 3 | 3 | 0 | 0 | 0 | 0 |  |
| puzzle_dump/state_0008_20260702_124637_968.json | puzzle_dump_state_0008_20260702_124637_968 | written | 3 | 2 | 2 | 0 | 0 | 0 | 0 |  |
| puzzle_dump/state_0009_20260702_124155_801.json | puzzle_dump_state_0009_20260702_124155_801 | written | 4 | 3 | 3 | 0 | 0 | 0 | 0 |  |
| puzzle_dump/state_0009_20260702_124659_435.json | puzzle_dump_state_0009_20260702_124659_435 | written | 2 | 1 | 1 | 0 | 0 | 0 | 0 |  |
| puzzle_dump/state_0010_20260702_124215_488.json | puzzle_dump_state_0010_20260702_124215_488 | written | 5 | 4 | 1 | 1 | 1 | 1 | 0 |  |
| puzzle_dump/state_0010_20260702_124715_094.json | puzzle_dump_state_0010_20260702_124715_094 | written | 5 | 4 | 3 | 0 | 0 | 0 | 0 |  |
| puzzle_dump/state_0011_20260702_124245_832.json | puzzle_dump_state_0011_20260702_124245_832 | written | 5 | 4 | 1 | 0 | 3 | 0 | 0 |  |
| puzzle_dump/state_0011_20260702_124732_358.json | puzzle_dump_state_0011_20260702_124732_358 | written | 4 | 3 | 3 | 0 | 0 | 0 | 0 |  |
| puzzle_dump/state_0012_20260702_124310_058.json | puzzle_dump_state_0012_20260702_124310_058 | written | 3 | 2 | 2 | 0 | 0 | 0 | 0 |  |
| puzzle_dump/state_0012_20260702_124749_409.json | puzzle_dump_state_0012_20260702_124749_409 | written | 3 | 2 | 2 | 0 | 0 | 0 | 0 |  |
| puzzle_dump/state_0013_20260702_124818_255.json | puzzle_dump_state_0013_20260702_124818_255 | written | 8 | 7 | 2 | 0 | 0 | 0 | 0 |  |
| puzzle_dump/state_0014_20260702_124832_691.json | puzzle_dump_state_0014_20260702_124832_691 | written | 6 | 5 | 1 | 0 | 0 | 0 | 0 |  |
| puzzle_dump/state_0015_20260702_124851_745.json | puzzle_dump_state_0015_20260702_124851_745 | written | 4 | 3 | 3 | 0 | 0 | 0 | 0 |  |
| puzzle_dump/state_0016_20260702_124906_783.json | puzzle_dump_state_0016_20260702_124906_783 | written | 4 | 3 | 3 | 0 | 0 | 0 | 0 |  |
| puzzle_dump/state_0017_20260702_124926_573.json | puzzle_dump_state_0017_20260702_124926_573 | written | 3 | 2 | 2 | 0 | 0 | 0 | 0 |  |
