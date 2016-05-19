
import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.util.Scanner;
import java.util.logging.Level;
import java.util.logging.Logger;

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/**
 *
 * @author rub account
 */
public class Format {
    public static void main(String[] args) throws UnsupportedEncodingException {
        try {
            Scanner sc = new Scanner(new File("absentCOGs on negative samples.csv"));
            PrintWriter f=new PrintWriter("absent.csv","UTF-8");
            
            while(sc.hasNext()){
                String s = sc.nextLine();
                if(!s.equals("del")){
                    f.println(s);
                }
            }
            f.close();
        } catch (FileNotFoundException ex) {
            Logger.getLogger(Format.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
    
}
