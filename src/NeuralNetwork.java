public class NeuralNetwork {
    private final Neuron[] hidden;
    private final Neuron[] output;
    private final double learningRate;

    public NeuralNetwork(int entrada, int oculto, int saida, double learningRate) {
        hidden = new Neuron[oculto];
        output = new Neuron[saida];
        this.learningRate = learningRate;

        for (int i = 0; i < oculto; i++)
            hidden[i] = new Neuron(entrada);

        for (int i = 0; i < saida; i++)
            output[i] = new Neuron(oculto);
    }

    public double[] feedforward(double[] input) {
        double[] hiddenOut = new double[hidden.length];
        for (int i = 0; i < hidden.length; i++)
            hiddenOut[i] = hidden[i].activate(input);

        double[] outputOut = new double[output.length];
        for (int i = 0; i < output.length; i++)
            outputOut[i] = output[i].activate(hiddenOut);

        return outputOut;
    }

    public void train(double[] input, double[] target) {
        // Feedforward
        double[] hiddenOut = new double[hidden.length];
        for (int i = 0; i < hidden.length; i++)
            hiddenOut[i] = hidden[i].activate(input);

        double[] outputOut = new double[output.length];
        for (int i = 0; i < output.length; i++)
            outputOut[i] = output[i].activate(hiddenOut);

        // Backpropagation (output layer)
        double[] outputDeltas = new double[output.length];
        for (int i = 0; i < output.length; i++) {
            double error = target[i] - outputOut[i];
            outputDeltas[i] = error * output[i].sigmoidDerivative();
        }

        // Backpropagation (hidden layer)
        double[] hiddenDeltas = new double[hidden.length];
        for (int i = 0; i < hidden.length; i++) {
            double error = 0;
            for (int j = 0; j < output.length; j++)
                error += outputDeltas[j] * output[j].weights[i];
            hiddenDeltas[i] = error * hidden[i].sigmoidDerivative();
        }

        // Update weights (output layer)
        for (int i = 0; i < output.length; i++)
            output[i].updateWeights(hiddenOut, outputDeltas[i], learningRate);

        // Update weights (hidden layer)
        for (int i = 0; i < hidden.length; i++)
            hidden[i].updateWeights(input, hiddenDeltas[i], learningRate);
    }
}
