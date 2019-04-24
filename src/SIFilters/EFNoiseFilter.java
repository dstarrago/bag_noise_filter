/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package SIFilters;

import java.util.Vector;
import keel.Algorithms.Preprocess.NoiseFilters.EnsembleFilter.EnsembleFilter;
import keel.Dataset.Attributes;
import weka.core.Instances;

/**
 *
 * @author Danel
 */
public class EFNoiseFilter extends KeelNoiseFilter {

  public EFNoiseFilter(Instances dataset) throws Exception {
    super(dataset);
  }
  
  public EFNoiseFilter(String dataFileName) throws Exception {
    super(dataFileName);
  }
  
//  System.out.println("\n\n\n**********************************************");
//  System.out.println("************* EJECUCION EF");
//  System.out.println("**********************************************");
  protected void runNoiseFilter(boolean[] cleanExample) {
    Attributes.clearAll();
    EnsembleFilter filter = new EnsembleFilter(dataFileName(), cleanExample);
    Vector res0 = filter.run();
    setDecisions((boolean[]) res0.get(0));
    setProposedClasses((int[]) res0.get(1));
    setNumNoisyExamples((Integer) res0.get(2));
  }
  
}
