import fasttext

model = fasttext.train_supervised(input="toutiao.train")
model.test("toutiao.valid", k=1)
# (76538, 0.8683660403982335, 0.8683660403982335)


model = fasttext.train_supervised(input="toutiao.train", epoch=25)
model.test("toutiao.valid", k=1)
# (76538, 0.8416472863152944, 0.8416472863152944)


model = fasttext.train_supervised(input="toutiao.train", lr=1.0)
model.test("toutiao.valid", k=1)
# (76538, 0.8567508949802712, 0.8567508949802712)

model = fasttext.train_supervised(input="toutiao.train", lr=0.01, epoch=3, wordNgrams=2)
# (76538, 0.4182758891008388, 0.4182758891008388)


model = fasttext.train_supervised(input="toutiao.train", lr=0.1, epoch=3, wordNgrams=2, bucket=200000, dim=50)
# (76538, 0.8696595155347671, 0.8696595155347671)


model = fasttext.train_supervised(input='toutiao.train', autotuneValidationFile='toutiao.valid', autotuneDuration=300)

