# Comparison with a measured motor: the U8 class

The motor in the MIT Mini Cheetah actuator is a U8-class outer-rotor motor
with 42 poles. Its published figures are compared here with what Pole-Craft
computes for it. The tests in `tests/studies/test_u8_comparison.py` hold
every number below, so the document stays true as the code changes.

## Sources

- **Katz 2018**: B. G. Katz, *A low cost modular actuator for dynamic
  robots*, SM thesis, MIT, 2018. The motor itself (section 2.2.1), torque
  against current (figure 2-3), cogging (section 2.5.2, figure 2-22), current
  control (section 2.4) and heating (section 2.5.3).
- **Urs 2022**: K. Urs, C. Enninful Adu, E. J. Rouse and T. Y. Moore,
  *Alternative metrics to select motors for quasi-direct drive actuators*,
  arXiv:2202.12365, 2022. Table 1 and figure 1, for the T-Motor U8 itself.

Only numbers are kept, in `tests/data/u8/measured.yaml`. The plotted curves
were read from the thesis's vector paths, not by eye. As a check, the
digitised cogging curve has an RMS of 0.0226 N m, and the thesis states
0.0228 N m.

## The design file

`tests/data/u8/u8_class_36s42p.yaml`. The sources state the air-gap
diameter (81 mm), the stack length (8.2 mm) and 21 pole pairs. The design
takes 36 slots, as in the U8. That agrees with the thesis's cogging, which
is mostly at 12 times the electrical frequency: LCM(36, 42) / 21 = 12.

**Everything else is an estimate.** That covers the gap, the magnet, the
teeth, the slots, the yokes and the wire. The steel and the magnet are the
library's placeholders.

The U8 is delta-connected (Urs 2022). The design describes the star winding
that behaves the same at the terminals: a delta of N turns per coil acts as
a star of N / √3 turns. The thesis's currents are line currents, which are
the phase currents of that star.

## Calibration

The turns per coil are fitted, not predicted. The measured torque from 0 to
10 A has a slope of 0.0791 N m/A. The drawn line starts at 0.02 N m, so the
slope is used rather than torque divided by current. The model matches this
slope with 6.8 turns. The design uses 7, which gives 0.0815 N m/A, 3 %
high; 6 turns would be 12 % low.

The fit absorbs everything that sets the magnet's flux per turn: the
placeholder magnet, the estimated gap, magnet thickness and arc. The turns
are therefore not a check of the model.

The two sources' torque constants differ by √3, which confirms the
convention. Urs 2022 gives 0.14 N m/A for the U8, divided by the current in
one delta phase. That current is the line current / √3. The ratio is 1.77
against √3 = 1.73.

## Checks

| Figure | Measured | Model | Verdict |
|---|---|---|---|
| Torque constant fall, low current to 40 A | 13 % (stated 12 %) | 2.3 % | **disagrees** |
| Inductance, star phase | about 30 µH (Katz); 23 µH (U8, 69 µH per delta phase) | 51 µH (50 in 2D, 1 from the end turns) | **disagrees, model 1.7 to 2.2 times higher** |
| Cogging, 12th electrical harmonic | 0.019 N m | 0.0007 to 0.0016 N m | not a check, see below |
| Turns per coil, as a delta | about 16, counted by eye | 7 × √3 = 12.1 | model 24 % lower |
| Phase resistance at 25 °C, star | 0.093 Ω (U8); 0.130 Ω (Katz, from the thermal figures) | 0.103 Ω | between them; the wire is estimated |
| Heat capacity | 32 J/K (winding, lumped) | 9.3 J/K (copper alone) | as intended, see below |

### Saturation

The motor's torque constant falls 13 % by 40 A, but the model's falls only
2.3 %. Several unknown inputs were changed one at a time, at 7 turns, to see
how far each could move the fall:

| Change | Fall at 40 A |
|---|---|
| None (the design as given) | 2.3 % |
| Real M270-35A steel in place of the placeholder | 3.0 % |
| Teeth 2.4 mm wide instead of 3.2 | 4.4 % |
| Stator yoke 1.5 mm instead of 2.5 | 2.7 % |
| Rotor yoke 1.5 mm instead of 2.5 | 2.7 % |
| 9 turns, with the magnet weakened to keep the same torque constant | 1.9 % |

With all the thin parts together and the real steel, the fall is 5.1 % at 7
turns and 6.7 % at 8. That is still about half of what was measured.

**This gap is not explained.** Some causes lie outside the 2D model:

- the motor's real steel, which may saturate sooner than M270-35A;
- the magnets warming during the test;
- the current measurement at high current;
- saturation in the end regions.

The design file was not changed to close the gap: nothing physical yet says
which of these it is.

### Inductance

The model's inductance is 1.7 times Katz's value and 2.2 times the U8's.
Both measurements come from fast transients: Katz used a current step
response, and Urs 2022 an LCR meter. At those frequencies, eddy currents in
the magnets and the rotor iron reduce the inductance; a static 2D field
solution does not include this.

The estimated slot opening and tooth tips set the slot leakage, which is
part of the inductance. Inductance also goes as the square of the turns,
and the turns were fitted through the placeholder magnet. Taking the U8's
counted 16 turns would make the gap larger, not smaller.

### Cogging

The model's cogging does not settle as the mesh is refined. Its peak is
0.0016, 0.0013 and 0.0007 N m at mesh sizes 0.5, 0.35 and 0.25, as found
for the reference motor. It also depends strongly on two estimated
dimensions:

| Slot opening | Magnet arc | Cogging peak |
|---|---|---|
| 1.5 mm | 0.85 | 0.0007 to 0.0013 N m |
| 3.0 mm | 0.85 | 0.009 N m |
| 1.5 mm | 0.75 | 0.006 N m |
| 1.5 mm | 0.95 | 0.0034 N m |
| 3.0 mm | 0.95 | 0.017 N m |

The measured slot harmonic, 0.019 N m, is within the reach of these
unknowns, so the order of magnitude is plausible but not checked.

Half of the measured cogging is a first harmonic of the electrical cycle
(0.018 N m), which comes from build tolerances. An ideal model cannot have
it.

### Heating

Katz measured a thermal resistance of 1.23 K/W and a thermal mass of 32 J/K
for the windings. These figures are self-consistent: 60 W gives
25 + 1.23 × 60 = 99 °C.

The stall-heating estimate (`polecraft.studies.stall_heating`) takes the
copper alone, with no heat leaving it. Its copper holds 9.3 J/K, 29 % of
Katz's lumped figure; the rest is the stator iron and insulation that the
lumped figure includes.

At 15.4 A from 25 °C:

- the worst phase reaches 100 °C in 8.3 s;
- the lumped 32 J/K, with no heat leaving it either, would take 65 s.

The estimate is on the safe side, as intended. It answers how long a hold
of a few seconds can last, not a continuous rating.

## Beside the reference design

Both are 36-slot, 42-pole outer-rotor motors with placeholder materials,
solved the same way.

| | Reference design | U8 model |
|---|---|---|
| Rotor outer diameter | 90 mm | 90.5 mm |
| Stack length | 12 mm | 8.2 mm |
| Turns per coil | 20 | 7 |
| Torque constant at 5 A | 0.321 N m/A | 0.0815 N m/A |
| Torque constant per turn and metre of stack | 1.34 | 1.42 |
| Torque constant fall at 40 A | 26 % | 2.3 % |
| d-axis inductance | 633 µH | 51 µH |
| Phase resistance | 0.956 Ω at 100 °C | 0.103 Ω at 25 °C |
| End factor (E8) | 0.996 | 0.995 |

Per turn and per metre of stack, the two give torque within 6 % of each
other, so the reference design is of the U8's class. Its larger fall at
40 A comes from 20 turns against 7: the same current drives almost three
times the ampere-turns.

## What this means

The model reproduces the torque constant only through a fitted turn count.
It understates saturation by a large and unexplained margin. It overstates
inductance by about twice, probably partly because the measurements are
transient ones.

Until real steel and magnet data and measured dimensions of one motor are
available, treat the model's figures as follows:

- **At low current:** a design comparison, not a prediction.
- **Torque at peak current:** optimistic.
- **Inductance:** pessimistic for current-control bandwidth.
