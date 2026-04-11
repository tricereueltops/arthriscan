from models.motor_model import MotorModel

m = MotorModel()
print("Taps | Speed  | Poor Dex Prob | Dexterity Level")
print("-" * 55)
for taps in [3, 5, 8, 10, 15, 20, 25, 30, 40]:
    f = m.preprocess(taps, 5.0)
    r = m.predict(f)
    speed = f["tapping_speed"]
    prob = r["poor_dexterity_probability"]
    level = r["dexterity_level"]
    print(f" {taps:3d}  | {speed:5.1f}  |    {prob:.3f}     | {level}")
