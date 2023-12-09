# Choquet capacity networks for random point process classification and regression

<!-- CONTACT -->
## Contact
Santiago VELASCO-FORERO
[Personal Website](https://people.cmm.minesparis.psl.eu/users/velasco/)


## Abstract

> In this study, we propose a new methodology that leverages the Choquet capacity, a mathematical tool from the theory of capacities, to capture the intricate properties of point patterns. By incorporating a dilation model, we enhance the ability to represent the spatial arrangement and density variations in the Neyman-Scott point process accurately. To validate the effectiveness of our approach, extensive experiments were conducted on synthetic datasets simulating diverse spatial point patterns on cell. The dilation model applied on real data with the distance function. We achieved the accuracy of $90\%$ which out performed CNN. The method is far faster than the other deep learning methods.

## Getting the code

You can download a copy of all the files in the repository morpholayersby cloning the repository:

    git clone https://github.com/Jacobiano/ChoquetLayer.git



## Dependencies

Tensorflow 2.0>=


## Reproducing the results

train_capacity2 can be use to generate the paper results.

Note:
In the first run, you have to keep --generatedate argument to 1 to generate simulations. 


<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

The authors gratefully acknowledge the financial support of Institut Carnot (grant 220000496).

This work was granted access to the HPC resources of IDRIS under the allocation 2023-AD011012212R2 made by GENCI



## License

MIT License

Copyright (c) 2020 Santiago Velasco-Forero

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
