/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package MIFilters.instance;

import weka.core.Instances;
import SIFilters.KeelNoiseFilter;
import java.lang.reflect.Constructor;
import java.util.logging.Level;
import java.util.logging.Logger;
import weka.core.Instance;

/**
 * Multi-instance noise filter that works at the instance level.
 * 
 * It is a meta filter that runs iteratively the [SimpleInstanceNoiseFilter]
 * until a given threshold of noise detection is reached.
 * 
 * @author Danel
 */
public class GenericIterativeInstanceNoiseFilter extends AbstractInstanceNoiseFilter {
  
  private final double LOWEST_APPRECIABLE_NOISE_PERCENT = 1;
  private int numIterations;
  private boolean[] decisions;
  private Instances siDataset;
  private Class<? extends AbstractInstanceNoiseFilter> miFilterClass;

  public GenericIterativeInstanceNoiseFilter(Instances dataset, 
          Class<? extends AbstractInstanceNoiseFilter> miFilterClass,
          Class<? extends KeelNoiseFilter> siFilterClass) throws Exception {
    super(dataset, siFilterClass);
    if (miFilterClass == GenericIterativeInstanceNoiseFilter.class) {
      throw new Exception("Illegal instance noise filter class");
    }
    this.miFilterClass = miFilterClass;
  }
  
  @Override
  public void apply() {
    try {
      siDataset = toSimpleInstance(getMIDataset());
    } catch (Exception ex) {
      Logger.getLogger(GenericIterativeInstanceNoiseFilter.class.getName()).log(Level.SEVERE, null, ex);
    }
    decisions = new boolean[siDataset.numInstances()];
    Instances targetMIData = getMIDataset();
    int partialNumFilteredInstances = 0;
    int significantNoiseThreshold;
    int numFilteredInstances = 0;
    do {
      significantNoiseThreshold = (int)(targetMIData.numInstances() * 
              LOWEST_APPRECIABLE_NOISE_PERCENT / 100.0);
      Constructor<? extends AbstractInstanceNoiseFilter> cons;
      try {
        cons = miFilterClass.getConstructor(Instances.class, siFilterClass.getClass());
        AbstractInstanceNoiseFilter filter = cons.newInstance(targetMIData, 
              getSingleInstanceFilterClass());
        filter.apply();
        targetMIData = filter.getNewMIDataset();
        updateDecisions(filter.decisions());
        partialNumFilteredInstances = filter.numFiltered();
        numFilteredInstances += partialNumFilteredInstances;
      } catch (Exception ex) {
        Logger.getLogger(GenericIterativeInstanceNoiseFilter.class.getName()).log(Level.SEVERE, null, ex);
      }
    } while (partialNumFilteredInstances > significantNoiseThreshold);
    setNewMIDataset(targetMIData);
    setDecisions(decisions);
    setNumFiltered(numFilteredInstances);
    super.apply();
  }
  
  private void updateDecisions(boolean[] partialDecisions) {
    int partialNoiseCount = 0;
    for (int i = 0; i < partialDecisions.length; i++) {
      if (partialDecisions[i]) {
        int count = partialNoiseCount++; 
        int index = 0;
        while (count < i || decisions[index]) {
          if (!decisions[index]) {
            count++;
          }
          index++;
        }
        decisions[index] = true;
      }
    }
  }
  
  public Instance getInstance(int index) {
    return siDataset.get(index);
  }
  
  public int getNumIterations() {
    return numIterations;
  }
  
  
}
