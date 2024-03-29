//author: god bennett, 2023
//neural networ/xor

import java.util.ArrayList;
import java.util.Scanner;

public class RunNeuralNetwork
{
    //features
    private static Scanner userScanner = new Scanner ( System.in );
    private static NeuralNetwork neuralNetwork = new NeuralNetwork ( );
    
    
    public static void main ( String arguments [ ] )
    {
        doTraining ( );
        doTesting ( );
    }
    
    //testing
    public static void doTesting ( )
    {
        System.out.println ( "God Neural Network From Scratch Practice 2023/2/15" );
        System.out.println ( "1. Get neural network guess for xor input [0,1]" );
        System.out.println ( "2. Get neural network guess for xor input [1,0]" );
        System.out.println ( "3. Get neural network guess for xor input [1,1]" );
        System.out.println ( "4. Get neural network guess for xor input [0,0]" );
        System.out.println ( "5. Exit" );
        System.out.println ( "6. Select an option [0,1]" );
        
        int userOption = userScanner.nextInt ( );
        switch ( userOption )
        {
            case 1:
            {
                int [ ] inputs = new int [ ] { 0, 1 };
                neuralNetwork.doForwardPropagation ( inputs );
                double neuralNetworkOutcome = neuralNetwork.getOutcome ( );
                System.out.println ( "Neural network guess for [0, 1] : " + neuralNetworkOutcome );
                returnToTesting ( );
            }
            break;
            case 2:
            {
                int [ ] inputs = new int [ ] { 1, 0 };
                neuralNetwork.doForwardPropagation ( inputs );
                double neuralNetworkOutcome = neuralNetwork.getOutcome ( );
                System.out.println ( "Neural network guess for [1, 0] : " + neuralNetworkOutcome );
                returnToTesting ( );
            }
            break;
            case 3:
            {
                int [ ] inputs = new int [ ] { 1, 1 };
                neuralNetwork.doForwardPropagation ( inputs );
                double neuralNetworkOutcome = neuralNetwork.getOutcome ( );
                System.out.println ( "Neural network guess for [1, 1] : " + neuralNetworkOutcome );
                returnToTesting ( );
            }
            break;
            case 4:
            {
                int [ ] inputs = new int [ ] { 0, 0 };
                neuralNetwork.doForwardPropagation ( inputs );
                double neuralNetworkOutcome = neuralNetwork.getOutcome ( );
                System.out.println ( "Neural network guess for [0, 0] : " + neuralNetworkOutcome );
                returnToTesting ( );
            }
            break;
            case 5:
            {
                System.exit ( 0 );
            }
            break;
        }
    }
    public static void returnToTesting ( )
    {        
        userScanner.nextLine ( );
        userScanner.nextLine ( );
        System.out.println ( "\f" );
        doTesting ( );
    }
    
    //training
    public static void doTraining ( )
    {
        ArrayList <String> trainingData = getTrainingData ( );
        
        for ( int tDI = 0; tDI < trainingData.size ( ); tDI ++ )
        {
            //forward prop
            String inputLine = trainingData.get ( tDI ).split ( "::" ) [ 0 ];
            int inputPartA = Integer.parseInt ( inputLine.split ( "," ) [ 0 ] );
            int inputPartB = Integer.parseInt ( inputLine.split ( "," ) [ 1 ] );
            int inputs [ ] = new int [ ] { inputPartA, inputPartB };
            neuralNetwork.doForwardPropagation ( inputs );
            
            //backward prop
            int target = Integer.parseInt ( trainingData.get ( tDI ).split ( "::" ) [ 1 ] );
            neuralNetwork.doBackwardPropagation ( target );
        }
    }
    
    //training data
    public static ArrayList <String> getTrainingData ( )
    {
         ArrayList <String> returnValue = new ArrayList <String> ( );
         
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "0,0::0" );
         returnValue.add ( "1,0::1" );
         returnValue.add ( "0,1::1" );
         returnValue.add ( "1,1::0" );
         returnValue.add ( "1,1::0" );
         
         return  returnValue;
    }
}