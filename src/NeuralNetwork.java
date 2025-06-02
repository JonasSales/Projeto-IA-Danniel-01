import java.io.*;

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

    public void printBias() {
        System.out.println("\nBias da Camada Oculta:");
        for (int i = 0; i < hidden.length; i++) {
            System.out.printf("Neuron %d: %.6f%n", i, hidden[i].bias);
        }

        System.out.println("\nBias da Camada de Saída:");
        for (int i = 0; i < output.length; i++) {
            System.out.printf("Neuron %d: %.6f%n", i, output[i].bias);
        }
    }

    public void saveBias(String filePath) {
        try (PrintWriter writer = new PrintWriter(new FileWriter(filePath))) {
            for (Neuron h : hidden)
                writer.println(h.getBias());
            for (Neuron o : output)
                writer.println(o.getBias());
            System.out.println("Bias salvos em " + filePath);
        } catch (IOException e) {
            System.out.println("Erro ao salvar biases: " + e.getMessage());
        }
    }

    public void loadBias(String filePath) {
        try (BufferedReader reader = new BufferedReader(new FileReader(filePath))) {
            for (Neuron h : hidden)
                h.setBias(Double.parseDouble(reader.readLine()));
            for (Neuron o : output)
                o.setBias(Double.parseDouble(reader.readLine()));
            System.out.println("Bias carregados de " + filePath);
        } catch (IOException | NumberFormatException e) {
            System.out.println("Erro ao carregar biases: " + e.getMessage());
        }
    }


}
