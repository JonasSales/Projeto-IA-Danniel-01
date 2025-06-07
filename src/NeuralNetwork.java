import java.io.*;

public class NeuralNetwork {
    private final Neuron[] oculto;
    private final Neuron[] saida;
    private final double taxaDeAprendizado;
    private final int tamanhoCamadaDeEntrada; // Armazenar o tamanho da camada de entrada
    private final int tamanhoCamadaDeSaida; // Armazenar o tamanho da camada oculta

    public NeuralNetwork(int entrada, int oculto, int saida, double taxaDeAprendizado) {
        this.oculto = new Neuron[oculto];
        this.saida = new Neuron[saida];
        this.taxaDeAprendizado = taxaDeAprendizado;
        this.tamanhoCamadaDeEntrada = entrada; // Salva o tamanho da camada de entrada
        this.tamanhoCamadaDeSaida = oculto; // Salva o tamanho da camada oculta

        for (int i = 0; i < oculto; i++)
            this.oculto[i] = new Neuron(entrada);

        for (int i = 0; i < saida; i++)
            this.saida[i] = new Neuron(oculto);
    }

    public double[] feedforward(double[] entrada) {
        double[] hiddenOut = new double[oculto.length];
        for (int i = 0; i < oculto.length; i++)
            hiddenOut[i] = oculto[i].activate(entrada);

        double[] outputOut = new double[saida.length];
        for (int i = 0; i < saida.length; i++)
            outputOut[i] = saida[i].activate(hiddenOut);

        return outputOut;
    }

    public void train(double[] entrada, double[] faixa) {
        // Feedforward
        double[] hiddenOut = new double[oculto.length];
        for (int i = 0; i < oculto.length; i++)
            hiddenOut[i] = oculto[i].activate(entrada);

        double[] outputOut = new double[saida.length];
        for (int i = 0; i < saida.length; i++)
            outputOut[i] = saida[i].activate(hiddenOut);

        // Backpropagation (camada de saída)
        double[] outputDeltas = new double[saida.length];
        for (int i = 0; i < saida.length; i++) {
            double error = faixa[i] - outputOut[i];
            outputDeltas[i] = error * saida[i].derivadaSigmoid();
        }

        // Backpropagation (camada oculta)
        double[] hiddenDeltas = new double[oculto.length];
        for (int i = 0; i < oculto.length; i++) {
            double error = 0;
            for (int j = 0; j < saida.length; j++)
                error += outputDeltas[j] * saida[j].pesos[i];
            hiddenDeltas[i] = error * oculto[i].derivadaSigmoid();
        }

        // Atualiza os pesos da camada de saída
        for (int i = 0; i < saida.length; i++)
            saida[i].atualizarPesos(hiddenOut, outputDeltas[i], taxaDeAprendizado);

        // Atualiza os pesos da camada oculta
        for (int i = 0; i < oculto.length; i++)
            oculto[i].atualizarPesos(entrada, hiddenDeltas[i], taxaDeAprendizado);
    }

    //Método para imprimir os Bias da rede
    public void imprimirBias() {
        System.out.println("\nBias da Camada Oculta:");
        for (int i = 0; i < oculto.length; i++) {
            System.out.printf("Neuron %d: %.6f%n", i, oculto[i].bias);
        }

        System.out.println("\nBias da Camada de Saída:");
        for (int i = 0; i < saida.length; i++) {
            System.out.printf("Neuron %d: %.6f%n", i, saida[i].bias);
        }
    }

    //Método para imprimir os pesos da rede.
    public void imprimirPesos() {
        System.out.println("\nPesos da Camada Oculta:");
        for (int i = 0; i < oculto.length; i++) {
            System.out.printf("Neurônio Oculto %d:%n", i);
            double[] weights = oculto[i].getPesos();
            for (int j = 0; j < weights.length; j++) {
                System.out.printf("  Peso de entrada %d: %.6f%n", j, weights[j]);
            }
        }

        System.out.println("\nPesos da Camada de Saída:");
        for (int i = 0; i < saida.length; i++) {
            System.out.printf("Neurônio de Saída %d:%n", i);
            double[] weights = saida[i].getPesos();
            for (int j = 0; j < weights.length; j++) {
                System.out.printf("  Peso de entrada %d: %.6f%n", j, weights[j]);
            }
        }
    }


    //Metódo para salvar os dados do modelo.
    public void salvarDadosModelo(String filePath) {
        try (PrintWriter writer = new PrintWriter(new FileWriter(filePath))) {
            // Salva os biases e pesos da camada oculta
            for (Neuron h : oculto) {
                writer.println(h.getBias()); // Salva o bias
                for (double weight : h.getPesos()) {
                    writer.println(weight); // Salva cada peso
                }
            }
            // Salva os biases e pesos da camada de saída
            for (Neuron o : saida) {
                writer.println(o.getBias()); // Salva o bias
                for (double weight : o.getPesos()) {
                    writer.println(weight); // Salva cada peso
                }
            }
            System.out.println("Modelo da rede neural salvo em " + filePath);
        } catch (IOException e) {
            System.out.println("Erro ao salvar modelo: " + e.getMessage());
        }
    }

    //Metodo para carregar dados do modelo.
    public void carregarModelo(String filePath) {
        try (BufferedReader reader = new BufferedReader(new FileReader(filePath))) {
            String linha;
            // Carrega os biases e pesos da camada oculta
            for (Neuron h : oculto) {
                linha = reader.readLine();
                if (linha == null) throw new IOException("Arquivo incompleto para biases da camada oculta.");
                h.setBias(Double.parseDouble(linha)); // Carrega o bias

                double[] loadedWeights = new double[tamanhoCamadaDeEntrada]; // Cria array para os pesos da camada oculta
                for (int i = 0; i < tamanhoCamadaDeEntrada; i++) {
                    linha = reader.readLine();
                    if (linha == null) throw new IOException("Arquivo incompleto para pesos da camada oculta.");
                    loadedWeights[i] = Double.parseDouble(linha); // Carrega cada peso
                }
                h.setPesos(loadedWeights); // Atribui os pesos carregados
            }

            // Carrega os biases e pesos da camada de saída
            for (Neuron o : saida) {
                linha = reader.readLine();
                if (linha == null) throw new IOException("Arquivo incompleto para biases da camada de saída.");
                o.setBias(Double.parseDouble(linha)); // Carrega o bias

                double[] loadedWeights = new double[tamanhoCamadaDeSaida]; // Cria array para os pesos da camada de saída
                for (int i = 0; i < tamanhoCamadaDeSaida; i++) {
                    linha = reader.readLine();
                    if (linha == null) throw new IOException("Arquivo incompleto para pesos da camada de saída.");
                    loadedWeights[i] = Double.parseDouble(linha); // Carrega cada peso
                }
                o.setPesos(loadedWeights); // Atribui os pesos carregados
            }
            System.out.println("Modelo da rede neural carregado de " + filePath);
        } catch (IOException | NumberFormatException e) {
            System.out.println("Erro ao carregar modelo: " + e.getMessage());
        }
    }
}
