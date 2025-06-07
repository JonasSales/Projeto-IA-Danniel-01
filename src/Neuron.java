import java.util.Random;

public class Neuron {
    public double[] pesos; // Array para armazenar os pesos das conexões de entrada do neurônio.
    public double bias;      // O valor do bias do neurônio.
    private double saida;   // Armazena o valor de saída do neurônio após a ativação.

    // Construtor do neurônio.
    // entradas: o número de entradas que este neurônio receberá (e, portanto, o número de pesos).
    public Neuron(int entradas) {
        pesos = new double[entradas]; // Inicializa o array de pesos com o tamanho de entradas.
        Random rand = new Random();     // Cria uma nova instância de Random para gerar números aleatórios.

        // Calcula o limite para a inicialização dos pesos usando a inicialização He (ou Kaiming).
        // Isso ajuda a evitar o vanishing/exploding gradient em redes profundas com funções ReLU,
        // mas aqui é usado com sigmoid, e geralmente Xavier/Glorot é mais comum para sigmoid.
        // No entanto, é uma estratégia de inicialização de pesos.
        double limit = Math.sqrt(6.0 / entradas); // limite para a distribuição uniforme He

        // Inicializa os pesos com valores aleatórios dentro da faixa [-limit, +limit].
        for (int i = 0; i < entradas; i++) {
            pesos[i] = rand.nextDouble() * 2 * limit - limit; // valor entre -limit e +limit
        }

        // Inicializa o bias também com um valor aleatório na mesma faixa.
        bias = rand.nextDouble() * 2 * limit - limit; // bias também na faixa [-limit, limit]
    }

    // Método activate: calcula a saída do neurônio dado um conjunto de entradas.
    public double activate(double[] inputs) {
        double sum = bias; // Começa a soma com o valor do bias.
        // Itera sobre as entradas e pesos para calcular a soma ponderada.
        for (int i = 0; i < inputs.length; i++)
            sum += pesos[i] * inputs[i]; // Adiciona o produto da entrada pelo peso correspondente à soma.
        saida = sigmoid(sum); // Aplica a função de ativação sigmoide à soma e armazena o resultado como a saída.
        return saida; // Retorna a saída ativada do neurônio.
    }

    // Método updateWeights: ajusta os pesos e o bias do neurônio durante o backpropagation.
    // inputs: as entradas que foram usadas para calcular a saída atual.
    // delta: o erro ponderado (delta) calculado durante a fase de backpropagation.
    // taxaAprendizado: a taxa de aprendizado da rede.
    public void atualizarPesos(double[] inputs, double delta, double taxaAprendizado) {
        // Atualiza cada peso. O ajuste é proporcional à taxa de aprendizado, ao delta e à entrada correspondente.
        for (int i = 0; i < pesos.length; i++)
            pesos[i] += taxaAprendizado * delta * inputs[i];
        // Atualiza o bias. O ajuste é proporcional à taxa de aprendizado e ao delta.
        bias += taxaAprendizado * delta;
    }

    // Função de ativação sigmoide.
    public double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x)); // Calcula 1 / (1 + e^(-x)).
    }

    // Derivada da função sigmoide.
    // A derivada da sigmoide pode ser calculada eficientemente usando sua própria saída: output * (1 - output).
    public double derivadaSigmoid() {
        return saida * (1 - saida);
    }

    // Setter para o bias (usado para carregar biases de um arquivo).
    public void setBias(double bias) {
        this.bias = bias;
    }

    // Getter para o bias (usado para salvar biases em um arquivo).
    public double getBias() {
        return bias;
    }

    // NOVO: Getter para os pesos (usado para salvar pesos em um arquivo).
    public double[] getPesos() {
        return pesos;
    }

    // NOVO: Setter para os pesos (usado para carregar pesos de um arquivo).
    public void setPesos(double[] pesos) {
        this.pesos = pesos;
    }
}
