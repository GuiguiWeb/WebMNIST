PYTHON=python3

OUTPUT=docs/static/lenet5
EPOCHS=5


all: train_export clean

train_export:
	${PYTHON} -m webmnist -o ${OUTPUT} --epochs ${EPOCHS} -t -e

train:
	${PYTHON} -m webmnist -o ${OUTPUT} --epochs ${EPOCHS} -t

export:
	${PYTHON} -m webmnist -o ${OUTPUT} -e

clean:
	rm -rf ${OUTPUT}.pth ${OUTPUT}.onnx ${OUTPUT}.pb