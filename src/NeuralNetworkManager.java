import java.util.Scanner;

public class NeuralNetworkManager {

    private final NeuralNetwork rede;
    private final double[][] entradas;
    private final double[][] saidas;
    private final int epocas;

    public NeuralNetworkManager(int inputSize, int hiddenSize, int outputSize, double learningRate,
                                double[][] entradas, double[][] saidas, int epocas) {
        this.rede = new NeuralNetwork(inputSize, hiddenSize, outputSize, learningRate);
        this.entradas = entradas;
        this.saidas = saidas;
        this.epocas = epocas;
    }

    public void treinarRede() {
        System.out.println("Iniciando treinamento da rede neural...");
        Treinamento treinamento = new Treinamento();
        treinamento.treinar(rede, entradas, saidas, epocas); // treinar agora salva o modelo completo
        treinamento.resultados(rede, entradas);
    }

    // NOVO: Método para carregar o modelo completo (biases e pesos) da rede neural de um arquivo.
    public void carregarRede(String filePath) {
        System.out.println("Carregando modelo da rede neural de " + filePath + "...");
        rede.carregarModelo(filePath); // Chama o método loadModel da rede neural.
    }

    public void testarRede() {
        System.out.println("\nTestando a rede neural com dados de treinamento:");
        Treinamento treinamento = new Treinamento();
        treinamento.resultados(rede, entradas);
    }

    public void testarComValor() {
        Scanner scanner = new Scanner(System.in);
        System.out.print("Digite um valor de entrada (ex: 0.5): ");
        try {
            double valorDouble = scanner.nextDouble();
            double[] resultado = rede.feedforward(new double[]{valorDouble});
            System.out.printf("Entrada: %.2f -> Saída: ", valorDouble);
            for (double v : resultado) {
                System.out.print((v >= 0.5 ? 1 : 0) + " ");
            }
            System.out.println();
        } catch (java.util.InputMismatchException e) {
            System.out.println("Entrada inválida. Por favor, digite um número.");
        }
    }

    public void mostrarBiasEPesoModelo() {
        rede.imprimirBias();
        rede.imprimirPesos();
    }
}
