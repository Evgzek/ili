import java.util.Random;

public class XORPerceptron {
    private final int numInputs = 2;
    private final int numHiddenNodes = 2;
    private final int numOutputs = 1;
    private final double learningRate = 0.1;
    private final double[][] input = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    private final double[][] output = {{0}, {1}, {1}, {0}};
    private double[][] hiddenWeights = new double[numInputs][numHiddenNodes];
    private double[][] outputWeights = new double[numHiddenNodes][numOutputs];
    private double[] hiddenBiases = new double[numHiddenNodes];
    private double[] outputBiases = new double[numOutputs];
    private double[][] hiddenOutputs = new double[numInputs][numHiddenNodes];
    private double[][] outputOutputs = new double[numInputs][numOutputs];
    private Random random = new Random();

    public XORPerceptron() {
        initializeWeights();
    }

    private void initializeWeights() {
        for (int i = 0; i < numInputs; i++) {
            for (int j = 0; j < numHiddenNodes; j++) {
                hiddenWeights[i][j] = random.nextDouble() - 0.5;
            }
        }

        for (int i = 0; i < numHiddenNodes; i++) {
            for (int j = 0; j < numOutputs; j++) {
                outputWeights[i][j] = random.nextDouble() - 0.5;
            }
        }

        for (int i = 0; i < numHiddenNodes; i++) {
            hiddenBiases[i] = random.nextDouble() - 0.5;
        }

        for (int i = 0; i < numOutputs; i++) {
            outputBiases[i] = random.nextDouble() - 0.5;
        }
    }

    private double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    private double sigmoidDerivative(double x) {
        double s = sigmoid(x);
        return s * (1.0 - s);
    }

    public void train(int numEpochs) {
        for (int epoch = 0; epoch < numEpochs; epoch++) {
            double totalError = 0.0;

            for (int example = 0; example < input.length; example++) {
                // Forward propagation
                for (int j = 0; j < numHiddenNodes; j++) {
                    double activation = hiddenBiases[j];

                    for (int i = 0; i < numInputs; i++) {
                        activation += input[example][i] * hiddenWeights[i][j];
                    }

                    hiddenOutputs[example][j] = sigmoid(activation);
                }

                for (int k = 0; k < numOutputs; k++) {
                    double activation = outputBiases[k];

                    for (int j = 0; j < numHiddenNodes; j++) {
                        activation += hiddenOutputs[example][j] * outputWeights[j][k];
                    }

                    outputOutputs[example][k] = sigmoid(activation);
                }

                // Backpropagation
                double error = 0.0;

                for (int k = 0; k < numOutputs; k++) {
                    double delta = (output[example][k] - outputOutputs[example][k]) * sigmoidDerivative(outputOutputs[example][k]);
                    for (int j = 0; j < numHiddenNodes; j++) {
                        outputWeights[j][k] += learningRate * hiddenOutputs[example][j] * delta;
                    }

                    outputBiases[k] += learningRate * delta;
                    error += Math.pow(output[example][k] - outputOutputs[example][k], 2.0);
                }

                for (int j = 0; j < numHiddenNodes; j++) {
                    double delta = 0.0;

                    for (int k = 0; k < numOutputs; k++) {
                        delta += outputWeights[j][k] * (output[example][k] - outputOutputs[example][k]) * sigmoidDerivative(outputOutputs[example][k]);
                    }

                    delta *= sigmoidDerivative(hiddenOutputs[example][j]);

                    for (int i = 0; i < numInputs; i++) {
                        hiddenWeights[i][j] += learningRate * input[example][i] * delta;
                    }

                    hiddenBiases[j] += learningRate * delta;
                }

                totalError += error;
            }

            System.out.println("Epoch " + epoch + ": Total error = " + totalError);
        }
    }

    public double predict(double[] input) {
        double[] hiddenOutput = new double[numHiddenNodes];
        double outputOutput = 0.0;

        for (int j = 0; j < numHiddenNodes; j++) {
            double activation = hiddenBiases[j];

            for (int i = 0; i < numInputs; i++) {
                activation += input[i] * hiddenWeights[i][j];
            }

            hiddenOutput[j] = sigmoid(activation);
        }

        for (int k = 0; k < numOutputs; k++) {
            double activation = outputBiases[k];

            for (int j = 0; j < numHiddenNodes; j++) {
                activation += hiddenOutput[j] * outputWeights[j][k];
            }

            outputOutput = sigmoid(activation);
        }

        return outputOutput;
    }

    public static void main(String[] args) {
        XORPerceptron perceptron = new XORPerceptron();
        perceptron.train(1000);

        System.out.println("Predictions for XOR:");
        System.out.println("0 XOR 0 = " + perceptron.predict(new double[]{0, 0}));
        System.out.println("0 XOR 1 = " + perceptron.predict(new double[]{0, 1}));
        System.out.println("1 XOR 0 = " + perceptron.predict(new double[]{1, 0}));
        System.out.println("1 XOR 1 = " + perceptron.predict(new double[]{1, 1}));

        System.out.println("Predictions for AND:");
        System.out.println("0 AND 0 = " + perceptron.predict(new double[]{0, 0}));
        System.out.println("0 AND 1 = " + perceptron.predict(new double[]{0, 1}));
        System.out.println("1 AND 0 = " + perceptron.predict(new double[]{1, 0}));
        System.out.println("1 AND 1 = " + perceptron.predict(new double[]{1, 1}));

        System.out.println("Predictions for OR:");
        System.out.println("0 OR 0 = " + perceptron.predict(new double[]{0, 0}));
        System.out.println("0 OR 1 = " + perceptron.predict(new double[]{0, 1}));
        System.out.println("1 OR 0 = " + perceptron.predict(new double[]{1, 0}));
        System.out.println("1 OR 1 = " + perceptron.predict(new double[]{1, 1}));
    }
}
