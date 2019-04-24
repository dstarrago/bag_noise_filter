/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package SIFilters;

import java.io.File;
import java.util.Vector;
import java.util.logging.Level;
import java.util.logging.Logger;
import keel.Algorithms.Preprocess.NoiseFilters.RNG.RNG;
import keel.Dataset.Attributes;
import org.core.Fichero;
import weka.core.Instances;

/**
 *
 * @author Danel
 */
public class RNGNoiseFilter extends KeelNoiseFilter {

  static final private String TMP_FOLDER = System.getProperty("java.io.tmpdir") + "/";
//  static final private String TMP_FOLDER = System.getProperty("java.io.tmpdir");
  private String TRA_RNG;
  private String TST_RNG;
  private String CFG_RNG;

  public RNGNoiseFilter(Instances dataset) throws Exception {
    super(dataset);
  }
  
  public RNGNoiseFilter(String dataFileName) throws Exception {
    super(dataFileName);
  }
  
//  System.out.println("\n\n\n**********************************************");
//  System.out.println("************* EJECUCION RNG");
//  System.out.println("**********************************************");
  @Override
  protected void runNoiseFilter(boolean[] cleanExample) {
    try {
      TRA_RNG = String.format("%saux_tra_rng_%s.dat", TMP_FOLDER, timeStamp);
      TST_RNG = String.format("%saux_tst_rng_%s.dat", TMP_FOLDER, timeStamp);
      CFG_RNG = String.format("%sconfig_RNG_%s.txt", TMP_FOLDER, timeStamp);
      Attributes.clearAll();
      CreateConfigFileRNG(dataFileName(), dataFileName(), TRA_RNG, TST_RNG);
      RNG filter = new RNG (CFG_RNG);
      Vector res0 = filter.ejecutar(cleanExample);
      setDecisions((boolean[]) res0.get(0));
      setProposedClasses((int[]) res0.get(1));
      setNumNoisyExamples((Integer) res0.get(2));
      File tmpFile = new File(CFG_RNG);
      tmpFile.delete();
    } catch (Exception ex) {
      Logger.getLogger(RNGNoiseFilter.class.getName()).log(Level.SEVERE, null, ex);
    }
  }
  
  public void CreateConfigFileRNG(String tra_in, String tst_in, String tra_out, String tst_out) throws Exception{
    String content = "";
    content += 	"algorithm = Prototipe Selection based on Relative Neighbourhood Graph";
    content += "\ninputData = \"" + tra_in + "\" \"" + tst_in +"\"";
    content += "\noutputData = \"" + tra_out + "\" \"" + tst_out + "\"";
    content += "\n\nOrder of the Graph = 1st_order";
    content += "\nType of Selection = Edition";
    content += "\nDistance Function = Euclidean";
    Fichero.escribeFichero(CFG_RNG, content);
  }
  
}
