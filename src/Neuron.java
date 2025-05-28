import java.util.Random;

public class Neuron {
    public double[] weights;
    public double bias;
    private double output;

    public Neuron(int entradas) {
        weights = new double[entradas];
        Random rand = new Random();
        for (int i = 0; i < entradas; i++)
            weights[i] = rand.nextDouble() * 2 - 1;
        bias = rand.nextDouble() * 2 - 1;
    }

    public double activate(double[] inputs) {
        double sum = bias;
        for (int i = 0; i < inputs.length; i++)
            sum += weights[i] * inputs[i];
        output = sigmoid(sum);
        return output;
    }

    public void updateWeights(double[] inputs, double delta, double taxaAprendizado) {
        for (int i = 0; i < weights.length; i++)
            weights[i] += taxaAprendizado * delta * inputs[i];
        bias += taxaAprendizado * delta;
    }

    public double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    public double sigmoidDerivative() {
        return output * (1 - output);
    }
}
