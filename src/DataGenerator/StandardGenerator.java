/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package DataGenerator;

import java.util.ArrayList;

/**
 *
 * @author Danel
 */
public class StandardGenerator extends Generator {

  public StandardGenerator(int numDimensions, double meanNumInstances, 
          double stdDevNumInstances, int numBags, ArrayList<Concept> posConcepts, 
          ArrayList<Concept> negConcepts) throws Exception {
    super(numDimensions, meanNumInstances, stdDevNumInstances, numBags, posConcepts, negConcepts);
  }

  public StandardGenerator(int numDimensions, double meanNumInstances, 
          double stdDevNumInstances, int numBags) throws Exception {
    super(numDimensions, meanNumInstances, stdDevNumInstances, numBags);
  }

  public StandardGenerator(double meanNumInstances, 
          double stdDevNumInstances, int numBags) throws Exception {
    super(meanNumInstances, stdDevNumInstances, numBags);
  }

  @Override
  public ArrayList<double[]> getPosInstanceArrays() {
    ArrayList<double[]> posBag = new ArrayList<>();
    int numBagInstances = genNumBagInstances();
    int numPosInstances = random.nextInt(numBagInstances);
    /**
     * Sample positive instances
     */
    for (int i = 0; i < numPosInstances; i++) {
      if (posConcepts.size() == 1) {
        posBag.add(posConcepts.get(0).sample());
        registerPosInstance();
      } else {
        /**
         * Choose a positive concept
         */
        int posConceptIndex = random.nextInt(posConcepts.size());
        posBag.add(posConcepts.get(posConceptIndex).sample());
        registerPosInstance();
      }
    }
    /**
     * Sample negative instances
     */
    for (int i = 0; i < numBagInstances - numPosInstances; i++) {
      if (negConcepts.size() == 1) {
        posBag.add(negConcepts.get(0).sample());
        registerNegInstance();
      } else {
        /**
         * Choose a negative concept
         */
        int negConceptIndex = random.nextInt(negConcepts.size());
        posBag.add(negConcepts.get(negConceptIndex).sample());
        registerNegInstance();
      }
    }
    return posBag;
  }

  @Override
  public ArrayList<double[]> getNegInstanceArrays() {
    ArrayList<double[]> negBag = new ArrayList<>();
    int numBagInstances = genNumBagInstances();
    /**
     * Sample negative instances
     */
    for (int i = 0; i < numBagInstances; i++) {
      if (negConcepts.size() == 1) {
        negBag.add(negConcepts.get(0).sample());
        registerNegInstance();
      } else {
        /**
         * Choose a negative concept
         */
        int negConceptIndex = random.nextInt(negConcepts.size());
        negBag.add(negConcepts.get(negConceptIndex).sample());
        registerNegInstance();
      }
    }
    return negBag;
  }
  
}
