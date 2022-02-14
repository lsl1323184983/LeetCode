package com.lsl;

import com.lsl.one.ListNode;
import com.lsl.one.Solution;

import java.util.Arrays;

public class Main {

    public static void main(String[] args) {
	// write your code here
        int[] nums = new int[4];
        nums[0] = 2;
        nums[1] = 7;
        nums[2] = 11;
        nums[3] = 15;
        Solution solution = new Solution();
        int[] res1 = solution.twoSum(nums, 9);
        System.out.println(Arrays.toString(res1));

        int[] l1 = new int[]{9,9,9,9,9,9,9};
        int[] l2 = new int[]{9,9,9,9};
        ListNode head1 = new ListNode(-1);
        ListNode head2 = new ListNode(-1);
        ListNode L1 = head1;
        ListNode L2 = head2;
        for (int i : l1) {
            L1.next= new ListNode(i);
            L1 = L1.next;
        }
        for (int i : l2) {
            L2.next= new ListNode(i);
            L2 = L2.next;
        }
        ListNode res2 = solution.addTwoNumbers(head1.next,head2.next);
        while(res2!=null){
            System.out.println(res2.val);
            res2=res2.next;
        }
    }
}
