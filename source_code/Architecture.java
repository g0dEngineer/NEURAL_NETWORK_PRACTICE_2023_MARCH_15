//Author: God Bennett , 2023, march 15
//Title: Neural network from scratch/memory
import java.util.ArrayList;

public class Architecture extends ArrayList <Integer>
{
    //constructor
    public Architecture ( String description )
    {
        String [ ] parts = description.split ( "," );
        
        for ( int pI = 0; pI < parts.length; pI ++ )
        {
            add ( Integer.parseInt ( parts [ pI ] ) );
        }
    }
}