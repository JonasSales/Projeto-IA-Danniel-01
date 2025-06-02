public class Main {
    public static void main(String[] args) {
        double[][] entradas = {
                {0.0}, {0.14}, {0.28}, {0.42}, {0.57}, {0.71}, {0.85}, {1.0}
        };

        double[][] saidas = {
                {0,0,0}, {0,0,1}, {0,1,0}, {0,1,1},
                {1,0,0}, {1,0,1}, {1,1,0}, {1,1,1}
        };

        final int epocas = 2500;

        //Bons valores entradas * 2 e learning rate 2.53
        //2.875 ótimos resultados
        NeuralNetwork rede = new NeuralNetwork(1, entradas.length*2, 3, 2.875); // 1 entrada, 25 ocultos, 3 saídas
        Treinamento treinamento = new Treinamento();

        treinamento.treinar(rede, entradas, saidas, epocas);
        treinamento.resultados(rede, entradas);


        rede.printBias();


        /*System.out.println("\nTestando valores de 0.00 até 1.00 (passo de 0.01):");
        for (int i = 0; i <= 100; i++) {
            double entrada = i / 100.0;
            double[] resultado = rede.feedforward(new double[]{entrada});
            System.out.printf("Entrada: %.2f -> Saída: ", entrada);
            for (double v : resultado)
                System.out.print((v >= 0.5 ? 1 : 0) + " ");
            System.out.println();
        }*/

    }

}

