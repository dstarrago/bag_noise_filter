/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package MIFilters.instance;

import weka.core.Instances;
import SIFilters.KeelNoiseFilter;
import java.util.logging.Level;
import java.util.logging.Logger;
import weka.core.Instance;

/**
 * Multi-instance noise filter that works at the instance level.
 *
 * @author Danel
 */
public class SimpleInstanceNoiseFilter extends AbstractInstanceNoiseFilter {
  
  private Instances siDataset;
  
  public SimpleInstanceNoiseFilter(Instances dataset, Class<? extends KeelNoiseFilter> siFilterClass) {
    super(dataset, siFilterClass);
  }
  
  @Override
  public void apply() {
    try {
      siDataset = toSimpleInstance(getMIDataset());
      Instances sid = removeID(siDataset);
      KeelNoiseFilter siFilter = getSingleInstanceFilter(sid);
      Instances newSIDataset = new Instances(siDataset, 0);
      for (int i = 0; i < siDataset.numInstances(); i++) {
        if (!siFilter.decisions()[i]) {
          newSIDataset.add(siDataset.get(i));
        }
      }
      setNewMIDataset(toMultiInstance(newSIDataset));
      setDecisions(siFilter.decisions());
      setNumFiltered(siFilter.numNoisyExamples());
    } catch (Exception ex) {
      Logger.getLogger(SimpleInstanceNoiseFilter.class.getName()).log(Level.SEVERE, null, ex);
    }
    super.apply();
  }
  
  public Instance getInstance(int index) {
    return siDataset.get(index);
  }
  
}
