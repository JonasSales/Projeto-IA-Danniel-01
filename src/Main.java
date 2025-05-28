public class Main {
    public static void main(String[] args) {
        double[][] entradas = {
                {0.0}, {0.14}, {0.28}, {0.42}, {0.57}, {0.71}, {0.85}, {1.0}
        };

        double[][] saidas = {
                {0,0,0}, {0,0,1}, {0,1,0}, {0,1,1},
                {1,0,0}, {1,0,1}, {1,1,0}, {1,1,1}
        };

        NeuralNetwork rede = new NeuralNetwork(1, 25, 3); // 1 entrada, 20 neurônios ocultos, 3 saídas

        // Treinamento
        for (int i = 0; i < 20000; i++) {
            for (int j = 0; j < entradas.length; j++) {
                rede.train(entradas[j], saidas[j]);
            }
        }

        // Teste
        for (double[] entrada : entradas) {
            double[] resultado = rede.feedforward(entrada);
            System.out.printf("Entrada: %.2f => Saída: ", entrada[0]);
            for (double v : resultado)
                System.out.print((v >= 0.5 ? 1 : 0) + " ");
            System.out.println();
        }
    }
}
